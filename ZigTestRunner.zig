const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const BORDER = "=" ** 80;

var current_test: ?[]const u8 = null;
var workspace_root: []const u8 = "";

const Ansi = struct {
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const dim = "\x1b[2m";
    const red = "\x1b[31m";
    const green = "\x1b[32m";
    const cyan = "\x1b[36m";
    const magenta = "\x1b[35m";
};

const RunnerError = error{TestLeakedMemory};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const root_buffer = std.fs.cwd().realpathAlloc(allocator, ".") catch null;
    defer if (root_buffer) |buf| {
        allocator.free(buf);
        workspace_root = "";
    };
    if (root_buffer) |buf| {
        workspace_root = buf;
    }

    var env = Env.init(allocator);
    defer env.deinit(allocator);

    var slowest = SlowTracker.init(allocator, 5);
    defer slowest.deinit();

    var resolver = SourceResolver.init();
    defer resolver.deinit();

    var printer = Printer{};

    var pass: usize = 0;
    var fail: usize = 0;
    var skip: usize = 0;
    var leak: usize = 0;

    for (builtin.test_functions) |t| {
        if (isSetup(t)) {
            t.func() catch |err| {
                printer.ensureLine();
                std.debug.print("error: setup {s} failed: {s}\n", .{ t.name, @errorName(err) });
                return err;
            };
        }
    }

    test_loop: for (builtin.test_functions) |t| {
        if (isSetup(t) or isTeardown(t)) continue;

        const is_unnamed = isUnnamed(t);
        if (env.filter) |f| {
            if (!is_unnamed and std.mem.indexOf(u8, t.name, f) == null) {
                continue;
            }
        }

        const friendly_name = friendlyTestName(t.name);
        var status = Status.pass;
        var failure: ?Failure = null;

        FailureNoteStore.clear();
        slowest.startTiming();
        current_test = friendly_name;
        std.testing.allocator_instance = .{};
        const result = t.func();
        current_test = null;

        const ns_taken = slowest.endTiming(friendly_name);

        if (result) |_| {
            status = .pass;
            pass += 1;
        } else |err| switch (err) {
            error.SkipZigTest => {
                status = .skip;
                skip += 1;
            },
            else => {
                status = .fail;
                fail += 1;
                const note = FailureNoteStore.take();
                failure = Failure{ .err = err, .trace = @errorReturnTrace(), .note = note };
            },
        }

        if (std.testing.allocator_instance.deinit() == .leak) {
            leak += 1;
            if (status != .fail) {
                status = .fail;
                fail += 1;
                failure = Failure{ .err = RunnerError.TestLeakedMemory, .note = "memory leak detected" };
            } else {
                if (failure) |*f| {
                    f.leaked = true;
                } else {
                    failure = Failure{ .err = RunnerError.TestLeakedMemory, .note = "memory leak detected", .leaked = true };
                }
            }
        }

        if (env.verbose) {
            printer.ensureLine();
            const label = switch (status) {
                .pass => "PASSED",
                .fail => "FAILED",
                .skip => "SKIP",
            };
            const ms = @as(f64, @floatFromInt(ns_taken)) / 1_000_000.0;
            std.debug.print("{s} {s} ({d:.2}ms)\n", .{ label, friendly_name, ms });
        }

        if (status == .fail) {
            if (failure) |f| {
                if (env.verbose) {
                    printer.ensureLine();
                }
                emitFailureBlock(friendly_name, f, allocator, &resolver, env.verbose);
                if (env.fail_first) break :test_loop;
            }
        }
    }

    for (builtin.test_functions) |t| {
        if (isTeardown(t)) {
            t.func() catch |err| {
                printer.ensureLine();
                std.debug.print("error: teardown {s} failed: {s}\n", .{ t.name, @errorName(err) });
                return err;
            };
        }
    }

    const total = pass + fail;
    printer.summary(pass, total, fail, skip, leak);

    if (env.verbose) {
        printer.ensureLine();
        try slowest.display();
        std.debug.print("\n", .{});
    }

    std.posix.exit(if (fail == 0) 0 else 1);
}

const Failure = struct {
    err: anyerror,
    trace: ?*std.builtin.StackTrace = null,
    note: ?[]const u8 = null,
    leaked: bool = false,
};

const FailureNoteStore = struct {
    var buffer: [256]u8 = undefined;
    var len: usize = 0;

    fn clear() void {
        len = 0;
    }

    fn set(message: []const u8) void {
        const count = @min(message.len, buffer.len);
        std.mem.copyForwards(u8, buffer[0..count], message[0..count]);
        len = count;
    }

    fn take() ?[]const u8 {
        if (len == 0) return null;
        const slice = buffer[0..len];
        len = 0;
        return slice;
    }
};

fn emitFailureBlock(
    friendly_name: []const u8,
    failure: Failure,
    allocator: Allocator,
    resolver: *SourceResolver,
    verbose: bool,
) void {
    std.debug.print("{s}X {s}{s}{s}\n", .{ Ansi.red, Ansi.cyan, friendly_name, Ansi.reset });

    const location = resolver.resolve(allocator, failure.trace);
    defer if (location) |loc| allocator.free(loc.file);
    const note_location = if (location == null) parseNoteLocation(failure.note) else null;

    if (location) |loc| {
        const display_path = shortenPath(loc.file);
        std.debug.print("{s}{s}:{d}{s}\n", .{ Ansi.dim, display_path, loc.line, Ansi.reset });
        printCodeFrame(allocator, loc);
    } else if (note_location) |nl| {
        const display_path = shortenPath(nl.file);
        std.debug.print("{s}{s}:{d}{s}\n", .{ Ansi.dim, display_path, nl.line, Ansi.reset });
        if (allocator.dupe(u8, nl.file) catch null) |copy| {
            defer allocator.free(copy);
            const loc = SourceResolver.ResolvedLocation{ .file = copy, .line = nl.line };
            printCodeFrame(allocator, loc);
        } else {
            std.debug.print("{s}    (source unavailable){s}\n\n", .{ Ansi.dim, Ansi.reset });
        }
    } else {
        std.debug.print("{s}unknown location{s}\n\n", .{ Ansi.dim, Ansi.reset });
    }

    var reason_buf: [256]u8 = undefined;
    const message = buildMessageParts(&reason_buf, failure);

    std.debug.print("{s}error:{s} {s}\n", .{ Ansi.red, Ansi.reset, message.heading });

    const has_details = message.details != null or failure.leaked;
    if (has_details) {
        if (message.details) |detail| {
            printDetailLines(detail);
        }
        if (failure.leaked) {
            std.debug.print("{s}Leak:{s} memory leak detected\n", .{ Ansi.cyan, Ansi.reset });
        }
    }

    std.debug.print("\n", .{});

    if (verbose) {
        if (failure.trace) |trace| {
            std.debug.dumpStackTrace(trace.*);
        }
    }
}

const MessageParts = struct {
    heading: []const u8,
    details: ?[]const u8 = null,
};

const NoteLocation = struct {
    file: []const u8,
    line: u64,
    column: u64,
};

fn buildMessageParts(buffer: []u8, failure: Failure) MessageParts {
    if (failure.note) |note| {
        return splitNote(note);
    }
    var stream = std.io.fixedBufferStream(buffer);
    const writer = stream.writer();
    writer.writeAll(@errorName(failure.err)) catch {};
    return .{ .heading = buffer[0..stream.pos], .details = null };
}

fn splitNote(note: []const u8) MessageParts {
    const trimmed = std.mem.trim(u8, note, " \r\n");
    if (trimmed.len == 0) {
        return .{ .heading = "assertion failed", .details = null };
    }
    if (std.mem.indexOfScalar(u8, trimmed, '\n')) |idx| {
        const heading = std.mem.trimRight(u8, trimmed[0..idx], "\r");
        const detail_start = idx + 1;
        const rest = std.mem.trimLeft(u8, trimmed[detail_start..], "\n\r");
        return .{
            .heading = if (heading.len == 0) "assertion failed" else heading,
            .details = if (rest.len == 0) null else rest,
        };
    }
    return .{ .heading = trimmed, .details = null };
}

fn parseNoteLocation(note_opt: ?[]const u8) ?NoteLocation {
    const note = note_opt orelse return null;
    const trimmed = std.mem.trim(u8, note, " \r\n");
    if (trimmed.len == 0) return null;
    const first_line_end = std.mem.indexOfScalar(u8, trimmed, '\n') orelse trimmed.len;
    const first_line = trimmed[0..first_line_end];

    const first_colon = std.mem.indexOfScalar(u8, first_line, ':') orelse return null;
    const second_rel = std.mem.indexOfScalar(u8, first_line[first_colon + 1 ..], ':') orelse return null;
    const second_colon = first_colon + 1 + second_rel;
    const third_rel = std.mem.indexOfScalar(u8, first_line[second_colon + 1 ..], ':') orelse return null;
    const third_colon = second_colon + 1 + third_rel;

    const file_slice = std.mem.trim(u8, first_line[0..first_colon], " ");
    const line_slice = std.mem.trim(u8, first_line[first_colon + 1 .. second_colon], " ");
    const column_slice = std.mem.trim(u8, first_line[second_colon + 1 .. third_colon], " ");

    const line = std.fmt.parseUnsigned(u64, line_slice, 10) catch return null;
    const column = std.fmt.parseUnsigned(u64, column_slice, 10) catch return null;

    return .{
        .file = file_slice,
        .line = line,
        .column = column,
    };
}

fn printDetailLines(detail: []const u8) void {
    var remaining = detail;
    while (remaining.len != 0) {
        const newline = std.mem.indexOfScalar(u8, remaining, '\n');
        const line = if (newline) |idx| blk: {
            const slice = std.mem.trimRight(u8, remaining[0..idx], "\r");
            remaining = if (idx + 1 < remaining.len) remaining[idx + 1 ..] else "";
            break :blk slice;
        } else blk: {
            const slice = std.mem.trimRight(u8, remaining, "\r");
            remaining = "";
            break :blk slice;
        };
        if (line.len == 0) {
            std.debug.print("\n", .{});
            continue;
        }
        if (std.mem.startsWith(u8, line, "Expected:")) {
            printDetailValue(line, "Expected:", Ansi.green);
        } else if (std.mem.startsWith(u8, line, "Received:")) {
            printDetailValue(line, "Received:", Ansi.red);
        } else if (std.mem.startsWith(u8, line, "Tolerance:")) {
            std.debug.print("{s}{s}{s}\n", .{ Ansi.cyan, line, Ansi.reset });
        } else {
            std.debug.print("{s}\n", .{line});
        }
    }
}

fn printDetailValue(line: []const u8, label: []const u8, color: []const u8) void {
    const prefix = line[0..label.len];
    const rest = line[label.len..];
    const trimmed = std.mem.trimLeft(u8, rest, " ");
    const leading_len = rest.len - trimmed.len;
    const leading = rest[0..leading_len];
    std.debug.print("{s}{s}", .{ prefix, leading });
    if (trimmed.len == 0) {
        std.debug.print("\n", .{});
        return;
    }
    std.debug.print("{s}{s}{s}\n", .{ color, trimmed, Ansi.reset });
}

fn shortenPath(path: []const u8) []const u8 {
    if (workspace_root.len == 0 or path.len <= workspace_root.len) {
        return path;
    }
    if (std.mem.startsWith(u8, path, workspace_root)) {
        return stripLeadingSeparators(path[workspace_root.len..]);
    }
    return path;
}

fn stripLeadingSeparators(path: []const u8) []const u8 {
    var start: usize = 0;
    while (start < path.len) : (start += 1) {
        const c = path[start];
        if (c != '/' and c != '\\') break;
    }
    return path[start..];
}

fn printCodeFrame(allocator: Allocator, location: SourceResolver.ResolvedLocation) void {
    var file = std.fs.cwd().openFile(location.file, .{}) catch {
        std.debug.print("    (source unavailable)\n\n", .{});
        return;
    };
    defer file.close();

    const max_bytes: usize = 64 * 1024;
    const contents = file.readToEndAlloc(allocator, max_bytes) catch {
        std.debug.print("    (source unavailable)\n\n", .{});
        return;
    };
    defer allocator.free(contents);

    const line_u: usize = if (location.line == 0) 1 else @intCast(location.line);
    const start_line = line_u;
    const end_line = line_u;
    const width = digitCount(end_line);

    var offset: usize = 0;
    var current_line: usize = 1;
    while (offset <= contents.len and current_line <= end_line) : (current_line += 1) {
        const bounds = nextLine(contents, offset);
        if (current_line >= start_line and current_line <= end_line) {
            printNumberedLine(current_line, width, bounds.line);
            if (current_line == end_line) {
                const caret_col = caretColumn(bounds.line);
                printCaret(width, caret_col);
            }
        }
        if (bounds.next >= contents.len) break;
        offset = bounds.next;
    }
}

fn digitCount(value: usize) usize {
    var count: usize = 1;
    var v = value;
    while (v >= 10) : (v /= 10) {
        count += 1;
    }
    return count;
}

const LineBounds = struct {
    line: []const u8,
    next: usize,
};

fn nextLine(contents: []const u8, start: usize) LineBounds {
    if (start >= contents.len) return .{ .line = "", .next = contents.len };
    var end = start;
    while (end < contents.len and contents[end] != '\n') {
        end += 1;
    }
    var slice = contents[start..end];
    if (slice.len > 0 and slice[slice.len - 1] == '\r') {
        slice = slice[0 .. slice.len - 1];
    }
    const next_index = if (end >= contents.len) contents.len else end + 1;
    return .{ .line = slice, .next = next_index };
}

fn printNumberedLine(line_number: usize, width: usize, text: []const u8) void {
    const digits_needed = if (width > 0) width else 1;
    var num_buf: [32]u8 = undefined;
    const rendered = std.fmt.bufPrint(&num_buf, "{d}", .{line_number}) catch return;
    const pad = if (digits_needed > rendered.len) digits_needed - rendered.len else 0;

    std.debug.print("{s}", .{Ansi.dim});
    if (pad > 0) printSpaces(pad);
    std.debug.print("{s} | {s}{s}{s}\n", .{ rendered, Ansi.reset, text, Ansi.reset });
}

fn printCaret(width: usize, column: usize) void {
    const base = width + 3 + column;
    printSpaces(base);
    std.debug.print("{s}^{s}\n", .{ Ansi.red, Ansi.reset });
}

fn printSpaces(count: usize) void {
    var buffer: [64]u8 = undefined;
    var remaining = count;
    while (remaining > 0) {
        const chunk = @min(remaining, buffer.len);
        @memset(buffer[0..chunk], ' ');
        std.debug.print("{s}", .{buffer[0..chunk]});
        remaining -= chunk;
    }
}

fn caretColumn(line: []const u8) usize {
    // Find opening parenthesis for function call
    var paren_idx: ?usize = null;
    for (0..line.len) |i| {
        if (line[i] == '(') {
            paren_idx = i;
            break;
        }
    }

    // If we found a paren, find first non-whitespace after it
    if (paren_idx) |idx| {
        var col: usize = 0;
        for (0..idx + 1) |i| {
            const c = line[i];
            switch (c) {
                ' ' => col += 1,
                '\t' => col += 4 - (col % 4),
                else => col += 1,
            }
        }
        // Skip whitespace after opening paren
        for (line[idx + 1 ..]) |c| {
            switch (c) {
                ' ' => col += 1,
                '\t' => col += 4 - (col % 4),
                else => return col,
            }
        }
    }

    // Fallback: point to first non-whitespace
    var col: usize = 0;
    for (line) |c| {
        switch (c) {
            ' ' => col += 1,
            '\t' => col += 4 - (col % 4),
            else => return col,
        }
    }
    return col;
}

const Printer = struct {
    fn ensureLine(self: *Printer) void {
        _ = self;
    }

    fn summary(self: *Printer, pass: usize, total: usize, fail: usize, skip: usize, leak: usize) void {
        if (pass == 0 and fail == 0 and skip == 0 and leak == 0) {
            return;
        }
        self.ensureLine();
        const label = if (fail == 0) "PASSED" else "FAILED";

        var extra_buf: [64]u8 = undefined;
        var extra_stream = std.io.fixedBufferStream(&extra_buf);
        var wrote = false;
        const writer = extra_stream.writer();

        if (fail > 0) {
            writer.print("failures {d}", .{fail}) catch {};
            wrote = true;
        }
        if (skip > 0) {
            writer.print("{s}skips {d}", .{ if (wrote) ", " else "", skip }) catch {};
            wrote = true;
        }
        if (leak > 0) {
            writer.print("{s}leaks {d}", .{ if (wrote) ", " else "", leak }) catch {};
            wrote = true;
        }

        const status: Status = if (fail == 0) .pass else .fail;
        const color_code = status.color();
        const reset = "\x1b[0m";

        if (wrote) {
            std.debug.print("{s}{s}{s} {d} / {d} ({s})\n", .{ color_code, label, reset, pass, total, extra_buf[0..extra_stream.pos] });
        } else {
            std.debug.print("{s}{s}{s} {d} / {d}\n", .{ color_code, label, reset, pass, total });
        }
    }
};

const Status = enum {
    pass,
    fail,
    skip,

    fn color(self: Status) []const u8 {
        return switch (self) {
            .pass => Ansi.green,
            .fail => Ansi.red,
            .skip => Ansi.cyan,
        };
    }
};

const SlowTracker = struct {
    const SlowestQueue = std.PriorityDequeue(TestInfo, void, compareTiming);
    max: usize,
    slowest: SlowestQueue,
    timer: std.time.Timer,

    fn init(allocator: Allocator, count: u32) SlowTracker {
        const timer = std.time.Timer.start() catch @panic("failed to start timer");
        var slowest = SlowestQueue.init(allocator, {});
        slowest.ensureTotalCapacity(count) catch @panic("OOM");
        return .{
            .max = count,
            .timer = timer,
            .slowest = slowest,
        };
    }

    const TestInfo = struct {
        ns: u64,
        name: []const u8,
    };

    fn deinit(self: SlowTracker) void {
        self.slowest.deinit();
    }

    fn startTiming(self: *SlowTracker) void {
        self.timer.reset();
    }

    fn endTiming(self: *SlowTracker, test_name: []const u8) u64 {
        var timer = self.timer;
        const ns = timer.lap();

        var slowest = &self.slowest;

        if (slowest.count() < self.max) {
            slowest.add(TestInfo{ .ns = ns, .name = test_name }) catch @panic("failed to track test timing");
            return ns;
        }

        const fastest_of_the_slow = slowest.peekMin() orelse unreachable;
        if (fastest_of_the_slow.ns > ns) {
            return ns;
        }

        _ = slowest.removeMin();
        slowest.add(TestInfo{ .ns = ns, .name = test_name }) catch @panic("failed to track test timing");
        return ns;
    }

    fn display(self: *SlowTracker) !void {
        var slowest = self.slowest;
        const count = slowest.count();
        std.debug.print("Slowest {d} test{s}: \n", .{ count, if (count != 1) "s" else "" });
        while (slowest.removeMinOrNull()) |info| {
            const ms = @as(f64, @floatFromInt(info.ns)) / 1_000_000.0;
            std.debug.print("  {d:.2}ms\t{s}\n", .{ ms, info.name });
        }
    }

    fn compareTiming(context: void, a: TestInfo, b: TestInfo) std.math.Order {
        _ = context;
        return std.math.order(a.ns, b.ns);
    }
};

const Env = struct {
    verbose: bool,
    fail_first: bool,
    filter: ?[]const u8,

    fn init(allocator: Allocator) Env {
        return .{
            .verbose = readEnvBool(allocator, "TEST_VERBOSE", false),
            .fail_first = readEnvBool(allocator, "TEST_FAIL_FIRST", false),
            .filter = readEnv(allocator, "TEST_FILTER"),
        };
    }

    fn deinit(self: Env, allocator: Allocator) void {
        if (self.filter) |f| {
            allocator.free(f);
        }
    }

    fn readEnv(allocator: Allocator, key: []const u8) ?[]const u8 {
        const v = std.process.getEnvVarOwned(allocator, key) catch |err| {
            if (err == error.EnvironmentVariableNotFound) {
                return null;
            }
            std.log.warn("failed to get env var {s} due to err {}", .{ key, err });
            return null;
        };
        return v;
    }

    fn readEnvBool(allocator: Allocator, key: []const u8, deflt: bool) bool {
        const value = readEnv(allocator, key) orelse return deflt;
        defer allocator.free(value);
        return std.ascii.eqlIgnoreCase(value, "true");
    }
};

const SourceResolver = struct {
    const ResolvedLocation = struct {
        file: []u8,
        line: u64,
    };

    info: ?*std.debug.SelfInfo,

    fn init() SourceResolver {
        return .{ .info = std.debug.getSelfDebugInfo() catch null };
    }

    fn deinit(self: *SourceResolver) void {
        _ = self;
    }

    fn resolve(self: *SourceResolver, allocator: Allocator, trace: ?*std.builtin.StackTrace) ?ResolvedLocation {
        const info = self.info orelse return null;
        const st = trace orelse return null;

        var best: ?struct {
            loc: ResolvedLocation,
            kind: FrameKind,
        } = null;

        const len = st.instruction_addresses.len;
        const available = @min(st.index, len);
        if (available == 0) return null;

        var checked: usize = 0;
        var frame_index: usize = if (st.index == 0) 0 else (st.index - 1) % len;
        while (checked < available) : (checked += 1) {
            const address = st.instruction_addresses[frame_index];
            if (address != 0) {
                const maybe_loc = resolveAddress(info, allocator, address - 1) orelse null;
                if (maybe_loc) |loc| {
                    const kind = classifyFile(loc.file);
                    if (best) |*b| {
                        if (@intFromEnum(kind) < @intFromEnum(b.kind)) {
                            allocator.free(b.loc.file);
                            b.* = .{ .loc = loc, .kind = kind };
                        } else {
                            allocator.free(loc.file);
                        }
                    } else {
                        best = .{ .loc = loc, .kind = kind };
                    }
                    if (kind == .project) break;
                }
            }

            frame_index = if (frame_index == 0) len - 1 else frame_index - 1;
        }

        if (best) |result| {
            return result.loc;
        }

        return null;
    }

    fn resolveAddress(info: *std.debug.SelfInfo, allocator: Allocator, address: usize) ?ResolvedLocation {
        const module = info.getModuleForAddress(address) catch return null;
        const symbol = module.getSymbolAtAddress(info.allocator, address) catch return null;
        defer if (symbol.source_location) |sl| info.allocator.free(sl.file_name);
        const source = symbol.source_location orelse return null;
        const file_copy = allocator.dupe(u8, source.file_name) catch return null;
        return .{
            .file = file_copy,
            .line = source.line,
        };
    }

    const FrameKind = enum(u2) {
        project,
        harness,
        stdlib,
    };

    fn classifyFile(path: []const u8) FrameKind {
        if (std.mem.indexOf(u8, path, "ZigTestRunner.zig") != null) {
            return .harness;
        }
        if (isStdPath(path)) {
            return .stdlib;
        }
        return .project;
    }

    fn isStdPath(path: []const u8) bool {
        if (std.mem.indexOf(u8, path, "/std/") != null or std.mem.indexOf(u8, path, "\\std\\") != null) {
            return true;
        }
        if (std.mem.startsWith(u8, path, "std/") or std.mem.startsWith(u8, path, "std\\")) {
            return true;
        }
        if (std.mem.indexOf(u8, path, "lib/std/") != null or std.mem.indexOf(u8, path, "lib\\std\\") != null) {
            return true;
        }
        return false;
    }
};

fn friendlyTestName(name: []const u8) []const u8 {
    var it = std.mem.splitScalar(u8, name, '.');
    while (it.next()) |value| {
        if (std.mem.eql(u8, value, "test")) {
            const rest = it.rest();
            return if (rest.len > 0) rest else name;
        }
    }
    return name;
}

pub export fn zigTestRunnerStoreFailureNote(ptr: [*]const u8, len: usize) void {
    if (len == 0) {
        FailureNoteStore.clear();
        return;
    }
    FailureNoteStore.set(ptr[0..len]);
}

pub const panic = std.debug.FullPanic(struct {
    pub fn panicFn(msg: []const u8, first_trace_addr: ?usize) noreturn {
        if (current_test) |ct| {
            std.debug.print("{s}\npanic running \"{s}\"\n{s}\n", .{ BORDER, ct, BORDER });
        }
        std.debug.defaultPanic(msg, first_trace_addr);
    }
}.panicFn);

fn isUnnamed(t: std.builtin.TestFn) bool {
    const marker = ".test_";
    const test_name = t.name;
    const index = std.mem.indexOf(u8, test_name, marker) orelse return false;
    _ = std.fmt.parseInt(u32, test_name[index + marker.len ..], 10) catch return false;
    return true;
}

fn isSetup(t: std.builtin.TestFn) bool {
    return std.mem.endsWith(u8, t.name, "tests:beforeAll");
}

fn isTeardown(t: std.builtin.TestFn) bool {
    return std.mem.endsWith(u8, t.name, "tests:afterAll");
}
