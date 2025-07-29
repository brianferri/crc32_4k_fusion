const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "crc32_4k_fusion",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);

    const asm_step = b.step("asm", "Emit assembly file");
    const awf = b.addWriteFiles();
    awf.step.dependOn(b.getInstallStep());
    // Path is relative to the cache dir in which it *would've* been placed in
    const asm_file_name = try std.fmt.allocPrint(b.allocator, "../../../zig-out/asm/crc32-asm.s", .{});
    _ = awf.addCopyFile(exe.getEmittedAsm(), asm_file_name);
    asm_step.dependOn(&awf.step);
}
