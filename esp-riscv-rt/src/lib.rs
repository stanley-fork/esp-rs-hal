//! Minimal startup/runtime for RISC-V CPUs from Espressif.
//!
//! ## Features
//!
//! This crate provides:
//!
//! - Before main initialization of the `.bss` and `.data` sections controlled by features
//! - `#[entry]` to declare the entry point of the program
//!
//! ## Feature Flags
#![doc = document_features::document_features!()]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/46717278")]
#![deny(missing_docs)]
#![no_std]

use core::arch::global_asm;

pub use riscv;
use riscv::register::{mcause, mtvec};
pub use riscv_rt_macros::{entry, pre_init};

pub use self::Interrupt as interrupt;

#[unsafe(export_name = "error: esp-riscv-rt appears more than once in the dependency graph")]
#[doc(hidden)]
pub static __ONCE__: () = ();

unsafe extern "C" {
    // Boundaries of the .bss section
    static mut _bss_end: u32;
    static mut _bss_start: u32;

    // Boundaries of the .data section
    static mut _data_end: u32;
    static mut _data_start: u32;

    // Initial values of the .data section (stored in Flash)
    static _sidata: u32;
}

/// Rust entry point (_start_rust)
///
/// Zeros bss section, initializes data section and calls main. This function
/// never returns.
///
/// # Safety
///
/// This function should not be called directly by the user, and should instead
/// be invoked by the runtime implicitly.
#[unsafe(link_section = ".init.rust")]
#[unsafe(export_name = "_start_rust")]
pub unsafe extern "C" fn start_rust(a0: usize, a1: usize, a2: usize) -> ! {
    unsafe {
        unsafe extern "Rust" {
            fn hal_main(a0: usize, a1: usize, a2: usize) -> !;

            fn __post_init();

            fn _setup_interrupts();

        }

        __post_init();

        _setup_interrupts();

        hal_main(a0, a1, a2);
    }
}

/// Registers saved in trap handler
#[derive(Debug, Default, Clone, Copy)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[repr(C)]
pub struct TrapFrame {
    /// Return address, stores the address to return to after a function call or
    /// interrupt.
    pub ra: usize,
    /// Temporary register t0, used for intermediate values.
    pub t0: usize,
    /// Temporary register t1, used for intermediate values.
    pub t1: usize,
    /// Temporary register t2, used for intermediate values.
    pub t2: usize,
    /// Temporary register t3, used for intermediate values.
    pub t3: usize,
    /// Temporary register t4, used for intermediate values.
    pub t4: usize,
    /// Temporary register t5, used for intermediate values.
    pub t5: usize,
    /// Temporary register t6, used for intermediate values.
    pub t6: usize,
    /// Argument register a0, typically used to pass the first argument to a
    /// function.
    pub a0: usize,
    /// Argument register a1, typically used to pass the second argument to a
    /// function.
    pub a1: usize,
    /// Argument register a2, typically used to pass the third argument to a
    /// function.
    pub a2: usize,
    /// Argument register a3, typically used to pass the fourth argument to a
    /// function.
    pub a3: usize,
    /// Argument register a4, typically used to pass the fifth argument to a
    /// function.
    pub a4: usize,
    /// Argument register a5, typically used to pass the sixth argument to a
    /// function.
    pub a5: usize,
    /// Argument register a6, typically used to pass the seventh argument to a
    /// function.
    pub a6: usize,
    /// Argument register a7, typically used to pass the eighth argument to a
    /// function.
    pub a7: usize,
}

/// Trap entry point rust (_start_trap_rust)
///
/// `scause`/`mcause` is read to determine the cause of the trap. XLEN-1 bit
/// indicates if it's an interrupt or an exception. The result is examined and
/// ExceptionHandler or one of the core interrupt handlers is called.
///
/// # Safety
///
/// This function should not be called directly by the user, and should instead
/// be invoked by the runtime implicitly.
#[unsafe(link_section = ".trap.rust")]
#[unsafe(export_name = "_start_trap_rust")]
pub unsafe extern "C" fn start_trap_rust(trap_frame: *const TrapFrame) {
    unsafe extern "C" {
        fn ExceptionHandler(trap_frame: &TrapFrame);
        fn DefaultHandler();
    }

    unsafe {
        let cause = mcause::read();

        if cause.is_exception() {
            ExceptionHandler(&*trap_frame)
        } else if cause.code() < __INTERRUPTS.len() {
            let h = &__INTERRUPTS[cause.code()];
            if h.reserved == 0 {
                DefaultHandler();
            } else {
                (h.handler)();
            }
        } else {
            DefaultHandler();
        }
    }
}

#[doc(hidden)]
#[unsafe(no_mangle)]
#[allow(unused_variables, non_snake_case)]
pub fn DefaultExceptionHandler(trap_frame: &TrapFrame) -> ! {
    loop {
        // Prevent this from turning into a UDF instruction
        // see rust-lang/rust#28728 for details
        continue;
    }
}

#[doc(hidden)]
#[unsafe(no_mangle)]
#[allow(unused_variables, non_snake_case)]
pub fn DefaultInterruptHandler() {
    loop {
        // Prevent this from turning into a UDF instruction
        // see rust-lang/rust#28728 for details
        continue;
    }
}

// Interrupts
#[doc(hidden)]
pub enum Interrupt {
    UserSoft,
    SupervisorSoft,
    MachineSoft,
    UserTimer,
    SupervisorTimer,
    MachineTimer,
    UserExternal,
    SupervisorExternal,
    MachineExternal,
}

unsafe extern "C" {
    fn UserSoft();
    fn SupervisorSoft();
    fn MachineSoft();
    fn UserTimer();
    fn SupervisorTimer();
    fn MachineTimer();
    fn UserExternal();
    fn SupervisorExternal();
    fn MachineExternal();
}

#[doc(hidden)]
pub union Vector {
    pub handler: unsafe extern "C" fn(),
    pub reserved: usize,
}

#[doc(hidden)]
#[unsafe(no_mangle)]
pub static __INTERRUPTS: [Vector; 12] = [
    Vector { handler: UserSoft },
    Vector {
        handler: SupervisorSoft,
    },
    Vector { reserved: 0 },
    Vector {
        handler: MachineSoft,
    },
    Vector { handler: UserTimer },
    Vector {
        handler: SupervisorTimer,
    },
    Vector { reserved: 0 },
    Vector {
        handler: MachineTimer,
    },
    Vector {
        handler: UserExternal,
    },
    Vector {
        handler: SupervisorExternal,
    },
    Vector { reserved: 0 },
    Vector {
        handler: MachineExternal,
    },
];

#[doc(hidden)]
#[unsafe(no_mangle)]
#[rustfmt::skip]
pub unsafe extern "Rust" fn default_post_init() {}

/// Default implementation of `_setup_interrupts` that sets `mtvec`/`stvec` to a
/// trap handler address.
#[doc(hidden)]
#[unsafe(no_mangle)]
#[rustfmt::skip]
pub unsafe extern "Rust" fn default_setup_interrupts() { unsafe {
    unsafe extern "C" {
        fn _start_trap();
    }

    mtvec::write(
        {
            let mut mtvec = mtvec::Mtvec::from_bits(0);
            mtvec.set_trap_mode(mtvec::TrapMode::Vectored);
            mtvec.set_address(_start_trap as usize);
            mtvec
        }
    );
}}

/// Parse cfg attributes inside a global_asm call.
macro_rules! cfg_global_asm {
    {@inner, [$($x:tt)*], } => {
        global_asm!{$($x)*}
    };
    (@inner, [$($x:tt)*], #[cfg($meta:meta)] $asm:literal, $($rest:tt)*) => {
        #[cfg($meta)]
        cfg_global_asm!{@inner, [$($x)* $asm,], $($rest)*}
        #[cfg(not($meta))]
        cfg_global_asm!{@inner, [$($x)*], $($rest)*}
    };
    {@inner, [$($x:tt)*], $asm:literal, $($rest:tt)*} => {
        cfg_global_asm!{@inner, [$($x)* $asm,], $($rest)*}
    };
    {$($asms:tt)*} => {
        cfg_global_asm!{@inner, [], $($asms)*}
    };
}

cfg_global_asm! {
    r#"
/*
    Entry point of all programs (_start).

    It initializes DWARF call frame information, the stack pointer, the
    frame pointer (needed for closures to work in start_rust) and the global
    pointer. Then it calls _start_rust.
*/

.section .init, "ax"
.global _start

_start:
    /* Jump to the absolute address defined by the linker script. */
    lui ra, %hi(_abs_start)
    jr %lo(_abs_start)(ra)

_abs_start:
    .option norelax
    .cfi_startproc
    .cfi_undefined ra
"#,
#[cfg(feature = "has-mie-mip")]
    r#"
    csrw mie, 0
    csrw mip, 0
"#,
    r#"
    la a0, _bss_start
    la a1, _bss_end
    bge a0, a1, 2f
    mv a3, x0
    1:
    sw a3, 0(a0)
    addi a0, a0, 4
    blt a0, a1, 1b
    2:
"#,
#[cfg(feature = "rtc-ram")]
    r#"
    la a0, _rtc_fast_bss_start
    la a1, _rtc_fast_bss_end
    bge a0, a1, 2f
    mv a3, x0
    1:
    sw a3, 0(a0)
    addi a0, a0, 4
    blt a0, a1, 1b
    2:
"#,
    // Zero .rtc_fast.persistent iff the chip just powered on
#[cfg(feature = "rtc-ram")]
    r#"
    mv a0, zero
    call rtc_get_reset_reason
    addi a1, zero, 1
    bne a0, a1, 2f
    la a0, _rtc_fast_persistent_start
    la a1, _rtc_fast_persistent_end
    bge a0, a1, 2f
    mv a3, x0
    1:
    sw a3, 0(a0)
    addi a0, a0, 4
    blt a0, a1, 1b
    2:
"#,
    r#"
    li  x1, 0
    li  x2, 0
    li  x3, 0
    li  x4, 0
    li  x5, 0
    li  x6, 0
    li  x7, 0
    li  x8, 0
    li  x9, 0
    li  x10,0
    li  x11,0
    li  x12,0
    li  x13,0
    li  x14,0
    li  x15,0
    li  x16,0
    li  x17,0
    li  x18,0
    li  x19,0
    li  x20,0
    li  x21,0
    li  x22,0
    li  x23,0
    li  x24,0
    li  x25,0
    li  x26,0
    li  x27,0
    li  x28,0
    li  x29,0
    li  x30,0
    li  x31,0

    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    // Check hart ID
    csrr t2, mhartid
    lui t0, %hi(_max_hart_id)
    add t0, t0, %lo(_max_hart_id)
    bgtu t2, t0, abort

    // Allocate stack
    la sp, _stack_start
    li t0, 4 // make sure stack start is in RAM
    sub sp, sp, t0
    andi sp, sp, -16 // Force 16-byte alignment

    // Set frame pointer
    add s0, sp, zero

    jal zero, _start_rust

    .cfi_endproc

/*
    Trap entry points (_start_trap, _start_trapN for N in 1..=31)

    The default implementation saves all registers to the stack and calls
    _start_trap_rust, then restores all saved registers before `mret`
*/
.section .trap, "ax"
.weak _start_trap  /* Exceptions call into _start_trap in vectored mode */
.weak _start_trap1
.weak _start_trap2
.weak _start_trap3
.weak _start_trap4
.weak _start_trap5
.weak _start_trap6
.weak _start_trap7
.weak _start_trap8
.weak _start_trap9
.weak _start_trap10
.weak _start_trap11
.weak _start_trap12
.weak _start_trap13
.weak _start_trap14
.weak _start_trap15
.weak _start_trap16
.weak _start_trap17
.weak _start_trap18
.weak _start_trap19
.weak _start_trap20
.weak _start_trap21
.weak _start_trap22
.weak _start_trap23
.weak _start_trap24
.weak _start_trap25
.weak _start_trap26
.weak _start_trap27
.weak _start_trap28
.weak _start_trap29
.weak _start_trap30
.weak _start_trap31
"#,
r#"
_start_trap: // Handle exceptions in vectored mode
    // move SP to some save place if it's pointing below the RAM
    // otherwise we won't be able to do anything reasonable
    // (since we don't have a working stack)
    //
    // most probably we will just print something and halt in this case
    // we actually can't do anything else
    csrw mscratch, t0
    la t0, _dram_origin
    bge sp, t0, 1f

    // use the reserved exception cause 14 to signal we detected a stack overflow
    li t0, 14
    csrw mcause, t0

    // set SP to the start of the stack
    la sp, _stack_start
    li t0, 4 // make sure stack start is in RAM
    sub sp, sp, t0
    andi sp, sp, -16 // Force 16-byte alignment

    1:
    csrr t0, mscratch
    // now SP is in RAM - continue

    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, _start_trap_rust_hal /* Load the HAL trap handler */
    j _start_trap_direct
_start_trap1:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt1
    j _start_trap_direct
_start_trap2:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt2
    j _start_trap_direct
_start_trap3:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt3
    j _start_trap_direct
_start_trap4:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt4
    j _start_trap_direct
_start_trap5:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt5
    j _start_trap_direct
_start_trap6:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt6
    j _start_trap_direct
_start_trap7:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt7
    j _start_trap_direct
_start_trap8:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt8
    j _start_trap_direct
_start_trap9:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt9
    j _start_trap_direct
_start_trap10:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt10
    j _start_trap_direct
_start_trap11:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt11
    j _start_trap_direct
_start_trap12:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt12
    j _start_trap_direct
_start_trap13:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt13
    j _start_trap_direct
_start_trap14:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt14
    j _start_trap_direct
_start_trap15:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt15
    j _start_trap_direct
_start_trap16:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt16
    j _start_trap_direct
_start_trap17:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt17
    j _start_trap_direct
_start_trap18:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt18
    j _start_trap_direct
_start_trap19:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt19
    j _start_trap_direct
_start_trap20:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt20
    j _start_trap_direct
_start_trap21:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt21
    j _start_trap_direct
_start_trap22:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt22
    j _start_trap_direct
_start_trap23:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt23
    j _start_trap_direct
_start_trap24:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt24
    j _start_trap_direct
_start_trap25:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt25
    j _start_trap_direct
_start_trap26:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt26
    j _start_trap_direct
_start_trap27:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt27
    j _start_trap_direct
_start_trap28:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt28
    j _start_trap_direct
_start_trap29:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt29
    j _start_trap_direct
_start_trap30:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt30
    j _start_trap_direct
_start_trap31:
    addi sp, sp, -16*4
    sw ra, 0(sp)
    la ra, interrupt31
    j _start_trap_direct
la ra, _start_trap_rust_hal /* this runs on exception, use regular fault handler */
_start_trap_direct:
"#,
r#"
    sw t0, 1*4(sp)
    sw t1, 2*4(sp)
    sw t2, 3*4(sp)
    sw t3, 4*4(sp)
    sw t4, 5*4(sp)
    sw t5, 6*4(sp)
    sw t6, 7*4(sp)
    sw a0, 8*4(sp)
    sw a1, 9*4(sp)
    sw a2, 10*4(sp)
    sw a3, 11*4(sp)
    sw a4, 12*4(sp)
    sw a5, 13*4(sp)
    sw a6, 14*4(sp)
    sw a7, 15*4(sp)

    # jump to handler loaded in direct handler
    add a0, sp, zero # load trap-frame address in a0
    jalr ra, ra # jump to label loaded in _start_trapX

    lw ra, 0*4(sp)
    lw t0, 1*4(sp)
    lw t1, 2*4(sp)
    lw t2, 3*4(sp)
    lw t3, 4*4(sp)
    lw t4, 5*4(sp)
    lw t5, 6*4(sp)
    lw t6, 7*4(sp)
    lw a0, 8*4(sp)
    lw a1, 9*4(sp)
    lw a2, 10*4(sp)
    lw a3, 11*4(sp)
    lw a4, 12*4(sp)
    lw a5, 13*4(sp)
    lw a6, 14*4(sp)
    lw a7, 15*4(sp)

    addi sp, sp, 16*4

    # SP was restored from the original SP
    mret

/* Make sure there is an abort when linking */
.section .text.abort
.globl abort
abort:
    j abort

/*
    Interrupt vector table (_vector_table)
*/

.section .trap, "ax"
.weak _vector_table
.type _vector_table, @function

.option push
.balign 0x100
.option norelax
.option norvc

_vector_table:
    j _start_trap
    j _start_trap1
    j _start_trap2
    j _start_trap3
    j _start_trap4
    j _start_trap5
    j _start_trap6
    j _start_trap7
    j _start_trap8
    j _start_trap9
    j _start_trap10
    j _start_trap11
    j _start_trap12
    j _start_trap13
    j _start_trap14
    j _start_trap15
    j _start_trap16
    j _start_trap17
    j _start_trap18
    j _start_trap19
    j _start_trap20
    j _start_trap21
    j _start_trap22
    j _start_trap23
    j _start_trap24
    j _start_trap25
    j _start_trap26
    j _start_trap27
    j _start_trap28
    j _start_trap29
    j _start_trap30
    j _start_trap31
.option pop
"#,
r#"
#this is required for the linking step, these symbols for in-use interrupts should always be overwritten by the user.
.section .trap, "ax"
// See https://github.com/esp-rs/esp-hal/issues/1326 and https://reviews.llvm.org/D98762
// and yes, this all has to go on one line... *sigh*.
.lto_discard interrupt1, interrupt2, interrupt3, interrupt4, interrupt5, interrupt6, interrupt7, interrupt8, interrupt9, interrupt10, interrupt11, interrupt12, interrupt13, interrupt14, interrupt15, interrupt16, interrupt17, interrupt18, interrupt19, interrupt20, interrupt21, interrupt22, interrupt23, interrupt24, interrupt25, interrupt26, interrupt27, interrupt28, interrupt29, interrupt30, interrupt31
.weak interrupt1
.weak interrupt2
.weak interrupt3
.weak interrupt4
.weak interrupt5
.weak interrupt6
.weak interrupt7
.weak interrupt8
.weak interrupt9
.weak interrupt10
.weak interrupt11
.weak interrupt12
.weak interrupt13
.weak interrupt14
.weak interrupt15
.weak interrupt16
.weak interrupt17
.weak interrupt18
.weak interrupt19
.weak interrupt20
.weak interrupt21
.weak interrupt22
.weak interrupt23
.weak interrupt24
.weak interrupt25
.weak interrupt26
.weak interrupt27
.weak interrupt28
.weak interrupt29
.weak interrupt30
.weak interrupt31
"#,
}
