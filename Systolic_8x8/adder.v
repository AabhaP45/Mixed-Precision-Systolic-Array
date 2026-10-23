// =========================================================================
// SHARED-DATAPATH SIMD FLOATING-POINT ADDER
//   FP32 / 2xFP16 / 4xFP8 (E4M3) / 8xFP4 (E2M1)
// =========================================================================
// One 32-bit datapath adds packed operands in whichever format
// mode_control selects:
//   2'b00 : 1x FP32  (E8M23)  -- lane 0 only, full 32-bit operand
//   2'b01 : 2x FP16  (E5M10)  -- lanes 0,1  (a[15:0], a[31:16])
//   2'b10 : 4x FP8   (E4M3)   -- lanes 0..3 (a[7:0],a[15:8],a[23:16],a[31:24])
//   2'b11 : 8x FP4   (E2M1)   -- lanes 0..7 (4 bits each, lane i = a[4i+3:4i])
//
// Datapath summary:
//   - Every lane is unpacked into sign / biased exponent / fraction and
//     zero-extended to the FP32 field widths, so the exponent logic is the
//     same for all formats.
//   - The exponent subtractor, the mantissa alignment shifter and the
//     mantissa adder are each ONE physical structure that is split into
//     independent lanes by mode-dependent breaks (carry breaks in the
//     adders, lane-boundary masking in the shifter).
//   - Subnormal inputs use an effective exponent of 1 and no hidden bit.
//     Subnormal FP32/FP16 results are flushed to signed zero when the
//     FTZ_EN parameter is 1; FP8/FP4 subnormal results are kept.
//   - Mantissas carry a single guard bit and there is no separate rounding
//     step: results are truncated after normalization.
//   - Special values: Inf/NaN inputs in FP32/FP16 and NaN inputs in FP8
//     bypass the arithmetic result (special_cases_simd). E4M3 has no Inf, so
//     FP8 overflow saturates to the largest finite value (448). FP4 has no
//     special encodings and also saturates on overflow.
//   - mode_control is carried down the pipeline with the data, so each stage
//     decodes the mode that belongs to the operands it is processing.
//
// The lane-break structure is described in the comments of each
// module, and synthesis maps the result onto shared carry-chain resources.
// =========================================================================


// =========================================================================
// 1. UNPACK MODULE
// =========================================================================
// Splits the packed 32-bit operands a and b into per-lane sign, biased
// exponent and fraction fields for the active format. Lanes that do not
// exist in the current mode are driven to zero. All fields are zero-extended
// to the FP32 widths (E32/M32) so later stages share one lane width.
//   FP32 : lane 0   = {s[31], e[30:23], f[22:0]}
//   FP16 : lane k   = {s, e[4:0], f[9:0]}  at a[16k+15:16k]
//   FP8  : lane k   = {s, e[3:0], f[2:0]}  at a[8k+7:8k]    (E4M3)
//   FP4  : lane k   = {s, e[1:0], f[0]}    at a[4k+3:4k]    (E2M1)
module unpack_simd #(
    parameter E32 = 8,  parameter M32 = 23,  // FP32: 8-bit exponent, 23-bit fraction
    parameter E16 = 5,  parameter M16 = 10,  // FP16: 5-bit exponent, 10-bit fraction
    parameter E8  = 4,  parameter M8  = 3,  // FP8 (E4M3): 4-bit exponent, 3-bit fraction
    parameter E4  = 2,  parameter M4  = 1  // FP4 (E2M1): 2-bit exponent, 1-bit fraction
)(
    input  [1:0]  mode_control,     // 00: FP32, 01: 2xFP16, 10: 4xFP8, 11: 8xFP4
    input  [31:0] a, b,

    // Per-lane sign / exponent / fraction of operands a and b (lanes 0..7).
    output reg signed_a_0, signed_b_0,
    output reg signed_a_1, signed_b_1,
    output reg signed_a_2, signed_b_2,
    output reg signed_a_3, signed_b_3,
    output reg signed_a_4, signed_b_4,
    output reg signed_a_5, signed_b_5,
    output reg signed_a_6, signed_b_6,
    output reg signed_a_7, signed_b_7,

    output reg [E32-1:0] exp_a_0, exp_b_0,
    output reg [E32-1:0] exp_a_1, exp_b_1,
    output reg [E32-1:0] exp_a_2, exp_b_2,
    output reg [E32-1:0] exp_a_3, exp_b_3,
    output reg [E32-1:0] exp_a_4, exp_b_4,
    output reg [E32-1:0] exp_a_5, exp_b_5,
    output reg [E32-1:0] exp_a_6, exp_b_6,
    output reg [E32-1:0] exp_a_7, exp_b_7,

    output reg [M32-1:0] frac_a_0, frac_b_0,
    output reg [M32-1:0] frac_a_1, frac_b_1,
    output reg [M32-1:0] frac_a_2, frac_b_2,
    output reg [M32-1:0] frac_a_3, frac_b_3,
    output reg [M32-1:0] frac_a_4, frac_b_4,
    output reg [M32-1:0] frac_a_5, frac_b_5,
    output reg [M32-1:0] frac_a_6, frac_b_6,
    output reg [M32-1:0] frac_a_7, frac_b_7
);

    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings

    // Combinational decode of the packed operands according to mode_control.
    always @(*) begin
        // Defaults: idle lanes are zeroed (harmless -- never selected downstream)
        signed_a_1 = 1'b0; signed_b_1 = 1'b0;
        signed_a_2 = 1'b0; signed_b_2 = 1'b0;
        signed_a_3 = 1'b0; signed_b_3 = 1'b0;
        signed_a_4 = 1'b0; signed_b_4 = 1'b0;
        signed_a_5 = 1'b0; signed_b_5 = 1'b0;
        signed_a_6 = 1'b0; signed_b_6 = 1'b0;
        signed_a_7 = 1'b0; signed_b_7 = 1'b0;
        exp_a_1 = {E32{1'b0}}; exp_b_1 = {E32{1'b0}};
        exp_a_2 = {E32{1'b0}}; exp_b_2 = {E32{1'b0}};
        exp_a_3 = {E32{1'b0}}; exp_b_3 = {E32{1'b0}};
        exp_a_4 = {E32{1'b0}}; exp_b_4 = {E32{1'b0}};
        exp_a_5 = {E32{1'b0}}; exp_b_5 = {E32{1'b0}};
        exp_a_6 = {E32{1'b0}}; exp_b_6 = {E32{1'b0}};
        exp_a_7 = {E32{1'b0}}; exp_b_7 = {E32{1'b0}};
        frac_a_1 = {M32{1'b0}}; frac_b_1 = {M32{1'b0}};
        frac_a_2 = {M32{1'b0}}; frac_b_2 = {M32{1'b0}};
        frac_a_3 = {M32{1'b0}}; frac_b_3 = {M32{1'b0}};
        frac_a_4 = {M32{1'b0}}; frac_b_4 = {M32{1'b0}};
        frac_a_5 = {M32{1'b0}}; frac_b_5 = {M32{1'b0}};
        frac_a_6 = {M32{1'b0}}; frac_b_6 = {M32{1'b0}};
        frac_a_7 = {M32{1'b0}}; frac_b_7 = {M32{1'b0}};

        signed_a_0 = 1'b0; signed_b_0 = 1'b0;
        exp_a_0 = {E32{1'b0}}; exp_b_0 = {E32{1'b0}};
        frac_a_0 = {M32{1'b0}}; frac_b_0 = {M32{1'b0}};

        case (mode_control)
            FP4: begin
                // E2M1, 8 lanes of 4 bits: lane i occupies bits [4i+3:4i] as
                // sign[3] | exp[2:1] | frac[0]
                signed_a_0=a[3];  signed_b_0=b[3];  exp_a_0={{(E32-E4){1'b0}},a[2:1]};   exp_b_0={{(E32-E4){1'b0}},b[2:1]};   frac_a_0={{(M32-M4){1'b0}},a[0]};  frac_b_0={{(M32-M4){1'b0}},b[0]};
                signed_a_1=a[7];  signed_b_1=b[7];  exp_a_1={{(E32-E4){1'b0}},a[6:5]};   exp_b_1={{(E32-E4){1'b0}},b[6:5]};   frac_a_1={{(M32-M4){1'b0}},a[4]};  frac_b_1={{(M32-M4){1'b0}},b[4]};
                signed_a_2=a[11]; signed_b_2=b[11]; exp_a_2={{(E32-E4){1'b0}},a[10:9]};  exp_b_2={{(E32-E4){1'b0}},b[10:9]};  frac_a_2={{(M32-M4){1'b0}},a[8]};  frac_b_2={{(M32-M4){1'b0}},b[8]};
                signed_a_3=a[15]; signed_b_3=b[15]; exp_a_3={{(E32-E4){1'b0}},a[14:13]}; exp_b_3={{(E32-E4){1'b0}},b[14:13]}; frac_a_3={{(M32-M4){1'b0}},a[12]}; frac_b_3={{(M32-M4){1'b0}},b[12]};
                signed_a_4=a[19]; signed_b_4=b[19]; exp_a_4={{(E32-E4){1'b0}},a[18:17]}; exp_b_4={{(E32-E4){1'b0}},b[18:17]}; frac_a_4={{(M32-M4){1'b0}},a[16]}; frac_b_4={{(M32-M4){1'b0}},b[16]};
                signed_a_5=a[23]; signed_b_5=b[23]; exp_a_5={{(E32-E4){1'b0}},a[22:21]}; exp_b_5={{(E32-E4){1'b0}},b[22:21]}; frac_a_5={{(M32-M4){1'b0}},a[20]}; frac_b_5={{(M32-M4){1'b0}},b[20]};
                signed_a_6=a[27]; signed_b_6=b[27]; exp_a_6={{(E32-E4){1'b0}},a[26:25]}; exp_b_6={{(E32-E4){1'b0}},b[26:25]}; frac_a_6={{(M32-M4){1'b0}},a[24]}; frac_b_6={{(M32-M4){1'b0}},b[24]};
                signed_a_7=a[31]; signed_b_7=b[31]; exp_a_7={{(E32-E4){1'b0}},a[30:29]}; exp_b_7={{(E32-E4){1'b0}},b[30:29]}; frac_a_7={{(M32-M4){1'b0}},a[28]}; frac_b_7={{(M32-M4){1'b0}},b[28]};
            end
            FP8: begin
                // E4M3: sign[7] | exp[6:3] (4b) | frac[2:0] (3b), per byte lane
                signed_a_0 = a[7];              signed_b_0 = b[7];
                exp_a_0    = {{(E32-E8){1'b0}}, a[6:3]};   exp_b_0 = {{(E32-E8){1'b0}}, b[6:3]};
                frac_a_0   = {{(M32-M8){1'b0}}, a[2:0]};   frac_b_0 = {{(M32-M8){1'b0}}, b[2:0]};

                signed_a_1 = a[15];             signed_b_1 = b[15];
                exp_a_1    = {{(E32-E8){1'b0}}, a[14:11]}; exp_b_1 = {{(E32-E8){1'b0}}, b[14:11]};
                frac_a_1   = {{(M32-M8){1'b0}}, a[10:8]};  frac_b_1 = {{(M32-M8){1'b0}}, b[10:8]};

                signed_a_2 = a[23];             signed_b_2 = b[23];
                exp_a_2    = {{(E32-E8){1'b0}}, a[22:19]}; exp_b_2 = {{(E32-E8){1'b0}}, b[22:19]};
                frac_a_2   = {{(M32-M8){1'b0}}, a[18:16]}; frac_b_2 = {{(M32-M8){1'b0}}, b[18:16]};

                signed_a_3 = a[31];             signed_b_3 = b[31];
                exp_a_3    = {{(E32-E8){1'b0}}, a[30:27]}; exp_b_3 = {{(E32-E8){1'b0}}, b[30:27]};
                frac_a_3   = {{(M32-M8){1'b0}}, a[26:24]}; frac_b_3 = {{(M32-M8){1'b0}}, b[26:24]};
            end
            FP16: begin
                // E5M10: sign[15] | exp[14:10] | frac[9:0]; lane 1 uses a[31:16]
                signed_a_0 = a[15];               signed_b_0 = b[15];
                exp_a_0    = {{(E32-E16){1'b0}}, a[14:10]}; exp_b_0 = {{(E32-E16){1'b0}}, b[14:10]};
                frac_a_0   = {{(M32-M16){1'b0}}, a[9:0]};   frac_b_0 = {{(M32-M16){1'b0}}, b[9:0]};

                signed_a_1 = a[31];               signed_b_1 = b[31];
                exp_a_1    = {{(E32-E16){1'b0}}, a[30:26]}; exp_b_1 = {{(E32-E16){1'b0}}, b[30:26]};
                frac_a_1   = {{(M32-M16){1'b0}}, a[25:16]}; frac_b_1 = {{(M32-M16){1'b0}}, b[25:16]};
            end
            default: begin // FP32
                // E8M23: sign[31] | exp[30:23] | frac[22:0]
                signed_a_0 = a[31];   signed_b_0 = b[31];
                exp_a_0    = a[30:23]; exp_b_0 = b[30:23];
                frac_a_0   = a[22:0];  frac_b_0 = b[22:0];
            end
        endcase
    end

endmodule


// =========================================================================
// 2. EXPONENT COMPARE MODULE (8 lanes, E=8 shared width)
// =========================================================================
// For each lane:
//   exp_*_notzero : exponent field is non-zero, i.e. the operand is normal
//                   and its implicit leading 1 is present (hidden bit).
//   eff_exp_*     : effective exponent; a zero exponent (zero/subnormal) is
//                   replaced by 1 so subnormals align like the smallest normal.
//   exp_a_larger  : eff_exp_a >= eff_exp_b (ties select operand a).
module exp_compare_simd #(
    parameter E = 8
)(
    input [E-1:0] exp_a_0, exp_b_0, exp_a_1, exp_b_1,
    input [E-1:0] exp_a_2, exp_b_2, exp_a_3, exp_b_3,
    input [E-1:0] exp_a_4, exp_b_4, exp_a_5, exp_b_5,
    input [E-1:0] exp_a_6, exp_b_6, exp_a_7, exp_b_7,

    output reg exp_a_notzero_0, exp_b_notzero_0, exp_a_larger_0,
    output reg exp_a_notzero_1, exp_b_notzero_1, exp_a_larger_1,
    output reg exp_a_notzero_2, exp_b_notzero_2, exp_a_larger_2,
    output reg exp_a_notzero_3, exp_b_notzero_3, exp_a_larger_3,
    output reg exp_a_notzero_4, exp_b_notzero_4, exp_a_larger_4,
    output reg exp_a_notzero_5, exp_b_notzero_5, exp_a_larger_5,
    output reg exp_a_notzero_6, exp_b_notzero_6, exp_a_larger_6,
    output reg exp_a_notzero_7, exp_b_notzero_7, exp_a_larger_7,

    output reg [E-1:0] eff_exp_a_0, eff_exp_b_0,
    output reg [E-1:0] eff_exp_a_1, eff_exp_b_1,
    output reg [E-1:0] eff_exp_a_2, eff_exp_b_2,
    output reg [E-1:0] eff_exp_a_3, eff_exp_b_3,
    output reg [E-1:0] eff_exp_a_4, eff_exp_b_4,
    output reg [E-1:0] eff_exp_a_5, eff_exp_b_5,
    output reg [E-1:0] eff_exp_a_6, eff_exp_b_6,
    output reg [E-1:0] eff_exp_a_7, eff_exp_b_7
);

    always @(*) begin
        // lane 0 (lanes 1..7 repeat the same logic)
        exp_a_notzero_0 = |exp_a_0; exp_b_notzero_0 = |exp_b_0;
        eff_exp_a_0 = exp_a_notzero_0 ? exp_a_0 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_0 = exp_b_notzero_0 ? exp_b_0 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_0 = (eff_exp_a_0 >= eff_exp_b_0);

        // lane 1
        exp_a_notzero_1 = |exp_a_1; exp_b_notzero_1 = |exp_b_1;
        eff_exp_a_1 = exp_a_notzero_1 ? exp_a_1 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_1 = exp_b_notzero_1 ? exp_b_1 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_1 = (eff_exp_a_1 >= eff_exp_b_1);

        // lane 2
        exp_a_notzero_2 = |exp_a_2; exp_b_notzero_2 = |exp_b_2;
        eff_exp_a_2 = exp_a_notzero_2 ? exp_a_2 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_2 = exp_b_notzero_2 ? exp_b_2 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_2 = (eff_exp_a_2 >= eff_exp_b_2);

        // lane 3
        exp_a_notzero_3 = |exp_a_3; exp_b_notzero_3 = |exp_b_3;
        eff_exp_a_3 = exp_a_notzero_3 ? exp_a_3 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_3 = exp_b_notzero_3 ? exp_b_3 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_3 = (eff_exp_a_3 >= eff_exp_b_3);

        // lane 4
        exp_a_notzero_4 = |exp_a_4; exp_b_notzero_4 = |exp_b_4;
        eff_exp_a_4 = exp_a_notzero_4 ? exp_a_4 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_4 = exp_b_notzero_4 ? exp_b_4 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_4 = (eff_exp_a_4 >= eff_exp_b_4);

        // lane 5
        exp_a_notzero_5 = |exp_a_5; exp_b_notzero_5 = |exp_b_5;
        eff_exp_a_5 = exp_a_notzero_5 ? exp_a_5 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_5 = exp_b_notzero_5 ? exp_b_5 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_5 = (eff_exp_a_5 >= eff_exp_b_5);

        // lane 6
        exp_a_notzero_6 = |exp_a_6; exp_b_notzero_6 = |exp_b_6;
        eff_exp_a_6 = exp_a_notzero_6 ? exp_a_6 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_6 = exp_b_notzero_6 ? exp_b_6 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_6 = (eff_exp_a_6 >= eff_exp_b_6);

        // lane 7
        exp_a_notzero_7 = |exp_a_7; exp_b_notzero_7 = |exp_b_7;
        eff_exp_a_7 = exp_a_notzero_7 ? exp_a_7 : {{E-1{1'b0}}, 1'b1};
        eff_exp_b_7 = exp_b_notzero_7 ? exp_b_7 : {{E-1{1'b0}}, 1'b1};
        exp_a_larger_7 = (eff_exp_a_7 >= eff_exp_b_7);
    end

endmodule


// =========================================================================
// 3. EXPONENT ABS-DIFFERENCE -- ONE physical 64-bit structure (8 bytes),
//    matching the mantissa adder's approach, widened to 8 lanes for FP4.
//    Exponent lanes never span more than one byte in any mode (FP32's
//    8-bit exponent fills exactly one byte; FP16/FP8/FP4's narrower
//    exponents each fit inside one byte with room to spare), so all eight
//    byte-subtractors are always independent -- but they are expressed
//    here as slices of a single 64-bit a64/b64 bus, not as eight separate
//    ports, so synthesis sees one composite word and can pack all eight
//    subtracts into contiguous physical resources instead of scattering
//    eight disconnected 8-bit subtractors across the die.
// =========================================================================
module mux_add_exp_simd (
    input [63:0] a64,   // byte-packed {exp_a_7,...,exp_a_0}
    input [63:0] b64,   // byte-packed {exp_b_7,...,exp_b_0}
    // a_greater_k: 1 when operand a has the larger (or equal) effective exponent
    // in lane k.
    input a_greater_0, a_greater_1, a_greater_2, a_greater_3,
    input a_greater_4, a_greater_5, a_greater_6, a_greater_7,
    output [63:0] diff64 // byte-packed {|d7|,...,|d0|}
);
    // Split the packed buses back into one byte per lane.
    wire [7:0] a0=a64[7:0],   b0=b64[7:0];
    wire [7:0] a1=a64[15:8],  b1=b64[15:8];
    wire [7:0] a2=a64[23:16], b2=b64[23:16];
    wire [7:0] a3=a64[31:24], b3=b64[31:24];
    wire [7:0] a4=a64[39:32], b4=b64[39:32];
    wire [7:0] a5=a64[47:40], b5=b64[47:40];
    wire [7:0] a6=a64[55:48], b6=b64[55:48];
    wire [7:0] a7=a64[63:56], b7=b64[63:56];

    // Route the larger/smaller operand per lane (pure muxing, no adder cost) --
    // result = bigger - smaller, always via invert + carry-in-of-1, so every
    // lane uses the exact same two's-complement subtract cell.
    wire [7:0] big0 = a_greater_0 ? a0 : b0,  sml0 = a_greater_0 ? b0 : a0;
    wire [7:0] big1 = a_greater_1 ? a1 : b1,  sml1 = a_greater_1 ? b1 : a1;
    wire [7:0] big2 = a_greater_2 ? a2 : b2,  sml2 = a_greater_2 ? b2 : a2;
    wire [7:0] big3 = a_greater_3 ? a3 : b3,  sml3 = a_greater_3 ? b3 : a3;
    wire [7:0] big4 = a_greater_4 ? a4 : b4,  sml4 = a_greater_4 ? b4 : a4;
    wire [7:0] big5 = a_greater_5 ? a5 : b5,  sml5 = a_greater_5 ? b5 : a5;
    wire [7:0] big6 = a_greater_6 ? a6 : b6,  sml6 = a_greater_6 ? b6 : a6;
    wire [7:0] big7 = a_greater_7 ? a7 : b7,  sml7 = a_greater_7 ? b7 : a7;

    // big - sml computed as big + ~sml + 1. Since big >= sml in every lane the
    // difference is non-negative, so the carry-out (bit 8) is discarded.
    wire [8:0] d0 = big0 + (~sml0) + 9'd1;
    wire [8:0] d1 = big1 + (~sml1) + 9'd1;
    wire [8:0] d2 = big2 + (~sml2) + 9'd1;
    wire [8:0] d3 = big3 + (~sml3) + 9'd1;
    wire [8:0] d4 = big4 + (~sml4) + 9'd1;
    wire [8:0] d5 = big5 + (~sml5) + 9'd1;
    wire [8:0] d6 = big6 + (~sml6) + 9'd1;
    wire [8:0] d7 = big7 + (~sml7) + 9'd1;

    // Repack the per-lane exponent differences.
    assign diff64 = {d7[7:0], d6[7:0], d5[7:0], d4[7:0], d3[7:0], d2[7:0], d1[7:0], d0[7:0]};
endmodule


// =========================================================================
// 4. EXPONENT DIFFERENCE WRAPPER
// =========================================================================
module exp_diff_simd #(
    parameter E = 8   // NOTE: this module packs each lane into a full byte,
                       // so it assumes E == 8 (true throughout this design --
                       // all lanes carry zero-extended 8-bit exponents).
)(
    input [E-1:0] exp_a_0, exp_b_0, exp_a_1, exp_b_1,
    input [E-1:0] exp_a_2, exp_b_2, exp_a_3, exp_b_3,
    input [E-1:0] exp_a_4, exp_b_4, exp_a_5, exp_b_5,
    input [E-1:0] exp_a_6, exp_b_6, exp_a_7, exp_b_7,
    input exp_a_larger_0, exp_a_larger_1, exp_a_larger_2, exp_a_larger_3,
    input exp_a_larger_4, exp_a_larger_5, exp_a_larger_6, exp_a_larger_7,

    output [E-1:0] exp_common_0, exp_diff_0,
    output [E-1:0] exp_common_1, exp_diff_1,
    output [E-1:0] exp_common_2, exp_diff_2,
    output [E-1:0] exp_common_3, exp_diff_3,
    output [E-1:0] exp_common_4, exp_diff_4,
    output [E-1:0] exp_common_5, exp_diff_5,
    output [E-1:0] exp_common_6, exp_diff_6,
    output [E-1:0] exp_common_7, exp_diff_7
);

    // Common exponent = the larger effective exponent of each lane, i.e. the
    // exponent the result is aligned to before normalization.
    assign exp_common_0 = exp_a_larger_0 ? exp_a_0 : exp_b_0;
    assign exp_common_1 = exp_a_larger_1 ? exp_a_1 : exp_b_1;
    assign exp_common_2 = exp_a_larger_2 ? exp_a_2 : exp_b_2;
    assign exp_common_3 = exp_a_larger_3 ? exp_a_3 : exp_b_3;
    assign exp_common_4 = exp_a_larger_4 ? exp_a_4 : exp_b_4;
    assign exp_common_5 = exp_a_larger_5 ? exp_a_5 : exp_b_5;
    assign exp_common_6 = exp_a_larger_6 ? exp_a_6 : exp_b_6;
    assign exp_common_7 = exp_a_larger_7 ? exp_a_7 : exp_b_7;

    // Byte-pack all eight lanes into ONE 64-bit "A" bus and ONE 64-bit "B"
    // bus (pure wiring -- each lane already occupies exactly one byte).
    wire [63:0] a64 = {exp_a_7, exp_a_6, exp_a_5, exp_a_4, exp_a_3, exp_a_2, exp_a_1, exp_a_0};
    wire [63:0] b64 = {exp_b_7, exp_b_6, exp_b_5, exp_b_4, exp_b_3, exp_b_2, exp_b_1, exp_b_0};
    wire [63:0] diff64;

    // Per-lane |exp_a - exp_b|: the right-shift amount applied to the smaller
    // operand's mantissa.
    mux_add_exp_simd u_mux_add_exp (
        .a64(a64), .b64(b64),
        .a_greater_0(exp_a_larger_0), .a_greater_1(exp_a_larger_1),
        .a_greater_2(exp_a_larger_2), .a_greater_3(exp_a_larger_3),
        .a_greater_4(exp_a_larger_4), .a_greater_5(exp_a_larger_5),
        .a_greater_6(exp_a_larger_6), .a_greater_7(exp_a_larger_7),
        .diff64(diff64)
    );

    // Unpack the difference bus back into per-lane values.
    assign exp_diff_0 = diff64[7:0];
    assign exp_diff_1 = diff64[15:8];
    assign exp_diff_2 = diff64[23:16];
    assign exp_diff_3 = diff64[31:24];
    assign exp_diff_4 = diff64[39:32];
    assign exp_diff_5 = diff64[47:40];
    assign exp_diff_6 = diff64[55:48];
    assign exp_diff_7 = diff64[63:56];

endmodule


// =========================================================================
// 5. SIGN XOR MODULE (8 lanes, gated by which lanes are active in each mode)
// =========================================================================
// signed_xor_k = sign_a_k ^ sign_b_k: 1 when lane k performs an effective
// subtraction (operand signs differ). Lanes that are inactive in the current
// mode are forced to 0:
//   lane 0    : always active
//   lane 1    : FP16, FP8, FP4 (any mode but FP32)
//   lanes 2,3 : FP8, FP4
//   lanes 4-7 : FP4 only
module signed_xor_simd (
    input [1:0] mode_control,

    input signed_a_0, signed_b_0, signed_a_1, signed_b_1,
    input signed_a_2, signed_b_2, signed_a_3, signed_b_3,
    input signed_a_4, signed_b_4, signed_a_5, signed_b_5,
    input signed_a_6, signed_b_6, signed_a_7, signed_b_7,

    output signed_xor_0, signed_xor_1, signed_xor_2, signed_xor_3,
    output signed_xor_4, signed_xor_5, signed_xor_6, signed_xor_7
);
    localparam FP32 = 2'b00, FP8 = 2'b10, FP4 = 2'b11;

    assign signed_xor_0 = signed_a_0 ^ signed_b_0;
    assign signed_xor_1 = (mode_control != FP32) ? (signed_a_1 ^ signed_b_1) : 1'b0;
    assign signed_xor_2 = (mode_control == FP8 || mode_control == FP4) ? (signed_a_2 ^ signed_b_2) : 1'b0;
    assign signed_xor_3 = (mode_control == FP8 || mode_control == FP4) ? (signed_a_3 ^ signed_b_3) : 1'b0;
    assign signed_xor_4 = (mode_control == FP4) ? (signed_a_4 ^ signed_b_4) : 1'b0;
    assign signed_xor_5 = (mode_control == FP4) ? (signed_a_5 ^ signed_b_5) : 1'b0;
    assign signed_xor_6 = (mode_control == FP4) ? (signed_a_6 ^ signed_b_6) : 1'b0;
    assign signed_xor_7 = (mode_control == FP4) ? (signed_a_7 ^ signed_b_7) : 1'b0;

endmodule


// =========================================================================
// 6. MANTISSA ADDER (behavioral; ONE segmented 32-bit adder shared by all modes)
// =========================================================================
// Each lane's mantissa (hidden bit + fraction + guard bit) sits at a fixed
// bit offset in a 32-bit bus. The per-format widths are WFP32 = 25, WFP16 = 12,
// WFP8 = 5 and WFP4 = 3 bits, and the adder adds one overflow bit above each
// lane's operand width:
//   FP32 uses 26 of 32 bits   (1 lane, full chain)
//   FP16 uses 13 of 16 bits   (2 lanes, 4-nibble chain each)
//   FP8  uses  6 of  8 bits   (4 lanes, 2-nibble chain each)
//   FP4  uses  4 of  4 bits   (8 lanes, 1 nibble each, exact fit)
//
// The adder is organized as 8 nibble (4-bit) segments with a mode-controlled
// carry break wherever a new independent lane starts.
//
// Nibble ownership by mode (does a NEW independent lane start at nibble k?):
//   nibble0 (bits  3:0 )  : always the start of lane 0
//   nibble1 (bits  7:4 )  : starts a new lane only in FP4 (else continues nibble0's lane)
//   nibble2 (bits 11:8 )  : starts a new lane in FP8 or FP4 (else continues)
//   nibble3 (bits 15:12)  : starts a new lane only in FP4 (else continues)
//   nibble4 (bits 19:16)  : starts a new lane in FP16, FP8 or FP4 (any mode but FP32)
//   nibble5 (bits 23:20)  : starts a new lane only in FP4 (else continues)
//   nibble6 (bits 27:24)  : starts a new lane in FP8 or FP4 (else continues)
//   nibble7 (bits 31:28)  : starts a new lane only in FP4 (else continues)
//
// base32 / shifted32 are nibble-aligned composite buses: base32 is built by
// shifter_prep_simd and shifted32 by shared_shifter32, each placing every
// lane's operand at the correct nibble offset for the current mode (pure
// wiring, no extra gates). The carry-break logic below keeps the lanes
// independent.
// -------------------------------------------------------------------------
module mux_add_mant_simd (
    input [1:0] mode_control,
    input signed_xor_0, signed_xor_1, signed_xor_2, signed_xor_3, // per-lane subtract flags
    input signed_xor_4, signed_xor_5, signed_xor_6, signed_xor_7,
    input [31:0] base32,     // nibble-aligned "larger magnitude" operand
    input [31:0] shifted32,  // nibble-aligned "aligned smaller magnitude" operand

    output [31:0] sum32      // nibble-aligned sum (sliced back to per-lane values by the caller)
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings

    // Does a NEW independent lane (and therefore a fresh carry-in) start at this nibble?
    wire n1_start = (mode_control == FP4);
    wire n2_start = (mode_control == FP8) || (mode_control == FP4);
    wire n3_start = (mode_control == FP4);
    wire n4_start = (mode_control != FP32);
    wire n5_start = (mode_control == FP4);
    wire n6_start = (mode_control == FP8) || (mode_control == FP4);
    wire n7_start = (mode_control == FP4);

    // Effective-subtract flag of the lane that owns each nibble. It drives the
    // B-operand inversion (sub_mask) and the +1 carry-in of the two's-complement
    // subtract. Each nibble takes the flag of whichever lane covers it in the
    // current mode: FP4 -> nibble k is lane k, FP8 -> lane k/2, FP16 -> lane k/4,
    // FP32 -> lane 0.
    wire sub0 = signed_xor_0;
    wire sub1 = (mode_control == FP4) ? signed_xor_1 : signed_xor_0;
    wire sub2 = (mode_control == FP4) ? signed_xor_2 :
                (mode_control == FP8) ? signed_xor_1 : signed_xor_0;
    wire sub3 = (mode_control == FP4) ? signed_xor_3 :
                (mode_control == FP8) ? signed_xor_1 : signed_xor_0;
    wire sub4 = (mode_control == FP4)  ? signed_xor_4 :
                (mode_control == FP8)  ? signed_xor_2 :
                (mode_control == FP16) ? signed_xor_1 : signed_xor_0;
    wire sub5 = (mode_control == FP4)  ? signed_xor_5 :
                (mode_control == FP8)  ? signed_xor_2 :
                (mode_control == FP16) ? signed_xor_1 : signed_xor_0;
    wire sub6 = (mode_control == FP4)  ? signed_xor_6 :
                (mode_control == FP8)  ? signed_xor_3 :
                (mode_control == FP16) ? signed_xor_1 : signed_xor_0;
    wire sub7 = (mode_control == FP4)  ? signed_xor_7 :
                (mode_control == FP8)  ? signed_xor_3 :
                (mode_control == FP16) ? signed_xor_1 : signed_xor_0;

    // ------------------------------------------------------------------
    // SINGLE 32-bit carry chain with mode-dependent lane breaks.
    //   * hmsb[k]: nibble k's MSB (bit 4k+3) is the MSB of a lane.
    //   * At every lane-MSB both addend bits are forced to the same value
    //     inj[k] = sub flag of the NEXT lane. Then the chain's sum bit there
    //     equals the incoming carry, and its carry-out is exactly inj[k]
    //     (= the +1 of two's-complement subtract for the next lane, or 0).
    //     The true MSB sum is restored with an XOR of (a ^ b_eff).
    //   * Lane 0's +1 for subtract is the chain carry-in (sub0).
    // ------------------------------------------------------------------
    // sub_v   : effective-subtract flag per nibble
    // start_v : nibble k starts a new lane (nibble 0 always does)
    // hmsb    : nibble k's MSB is the last bit of a lane (nibble k+1 starts a
    //           lane; nibble 7 always ends one)
    // inj     : value injected at each lane MSB = subtract flag of the next lane
    //           (0 for the top lane)
    wire [7:0] sub_v   = {sub7, sub6, sub5, sub4, sub3, sub2, sub1, sub0};
    wire [7:0] start_v = {n7_start, n6_start, n5_start, n4_start, n3_start, n2_start, n1_start, 1'b1};
    wire [7:0] hmsb    = {1'b1, start_v[7:1]};
    wire [7:0] inj     = {1'b0, sub_v[7:1]};

    // beff     : aligned operand, inverted in subtracting lanes (sub_mask
    //            replicates each nibble's sub flag across its 4 bits)
    // msb_sel  : mask of the lane-MSB bit positions
    // inj_bits : the injected value placed at those positions
    wire [31:0] sub_mask = {{4{sub7}},{4{sub6}},{4{sub5}},{4{sub4}},{4{sub3}},{4{sub2}},{4{sub1}},{4{sub0}}};
    wire [31:0] beff     = shifted32 ^ sub_mask;
    wire [31:0] msb_sel  = {hmsb[7],3'b0, hmsb[6],3'b0, hmsb[5],3'b0, hmsb[4],3'b0,
                            hmsb[3],3'b0, hmsb[2],3'b0, hmsb[1],3'b0, hmsb[0],3'b0};
    wire [31:0] inj_bits = {inj[7],3'b0, inj[6],3'b0, inj[5],3'b0, inj[4],3'b0,
                            inj[3],3'b0, inj[2],3'b0, inj[1],3'b0, inj[0],3'b0};

    // At each lane MSB both addends are forced to inj, so the carry chain
    // restarts cleanly for the next lane.
    wire [31:0] a_in = (base32 & ~msb_sel) | (inj_bits & msb_sel);
    wire [31:0] b_in = (beff   & ~msb_sel) | (inj_bits & msb_sel);
    // Single shared carry chain; sub0 is the +1 carry-in for lane 0. The final
    // carry-out (bit 32) is not used.
    wire [32:0] raw  = {1'b0, a_in} + {1'b0, b_in} + {32'b0, sub0};

    // At lane-MSB bits the chain output equals the incoming carry; XOR with
    // (base ^ beff) restores the true sum bit there.
    assign sum32 = raw[31:0] ^ ((base32 ^ beff) & msb_sel);

endmodule


// =========================================================================
// 6b. SHARED SEGMENTED BARREL SHIFTER (ONE physical 32-bit shift network,
//     reused by FP32 / 2xFP16 / 4xFP8 / 8xFP4 -- mirrors the adder's
//     carry-break idea, but applied to the shift-mux tree instead of the
//     carry chain)
// =========================================================================
// Standard barrel shifter = log2(32)=5 stages of 2:1 muxes (shift by
// 16,8,4,2,1). To make ONE such 32-bit network behave as a single 32-bit
// shifter, two independent 16-bit shifters, four independent 8-bit
// shifters, or eight independent 4-bit shifters, each stage's mux
// additionally checks whether the source bit it would pull in belongs to
// the SAME lane as the destination bit (given the current mode); if it
// would cross into a different lane it pulls in 0 instead of the
// neighbor's data -- exactly like a smaller, independent shifter would.
// This is the standard "segmented"/SIMD barrel-shifter construction.
// =========================================================================
module shared_shifter32 #(
    parameter E = 8,
    parameter M8 = 3   // FP8 mantissa width (hidden+frac+guard = M8+2); only width this module can't derive from E
)(
    input [1:0] mode_control,
    input [31:0] data_in,     // lane-aligned "operand to align" composite bus
    input [E-1:0] exp_diff_0, exp_diff_1, exp_diff_2, exp_diff_3,
    input [E-1:0] exp_diff_4, exp_diff_5, exp_diff_6, exp_diff_7,
    output [31:0] data_out
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    // Operand width per format: hidden bit + fraction + guard bit.
    localparam WFP32 = 25, WFP16 = 12, WFP8 = M8+2, WFP4 = 3;

    // Per-lane shift amount (5 bits covers 0..31, enough for the widest
    // format's max meaningful shift of 25; anything larger is caught by
    // the explicit force_zero compares below).
    wire [4:0] sh0 = exp_diff_0[4:0];
    wire [4:0] sh1 = exp_diff_1[4:0];
    wire [4:0] sh2 = exp_diff_2[4:0];
    wire [4:0] sh3 = exp_diff_3[4:0];
    wire [4:0] sh4 = exp_diff_4[4:0];
    wire [4:0] sh5 = exp_diff_5[4:0];
    wire [4:0] sh6 = exp_diff_6[4:0];
    wire [4:0] sh7 = exp_diff_7[4:0];

    // Per-bit-position lane id (0..7) for the current mode, and that
    // position's own shift amount -- pure structural dataflow (generate +
    // assign), no procedural loops/functions, so there's no ambiguity in
    // evaluation order.
    wire [2:0] lane_of   [0:31];
    wire [4:0] shamt_of  [0:31];

    genvar gi;
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_lane
            // Lane that bit gi belongs to: FP32 -> single lane; FP16 -> bit 4;
            // FP8 -> bits 4:3; FP4 -> bits 4:2.
            assign lane_of[gi] = (mode_control==FP32) ? 3'd0 :
                                  (mode_control==FP16) ? {2'b00, gi[4]} :
                                  (mode_control==FP8)  ? {1'b0, gi[4:3]} :
                                                          gi[4:2];
            // Shift amount of that lane.
            assign shamt_of[gi] = (lane_of[gi]==3'd0) ? sh0 :
                                   (lane_of[gi]==3'd1) ? sh1 :
                                   (lane_of[gi]==3'd2) ? sh2 :
                                   (lane_of[gi]==3'd3) ? sh3 :
                                   (lane_of[gi]==3'd4) ? sh4 :
                                   (lane_of[gi]==3'd5) ? sh5 :
                                   (lane_of[gi]==3'd6) ? sh6 : sh7;
        end
    endgenerate

    // 5 shift stages (16,8,4,2,1). Each stage pulls in the neighbor 2^k
    // positions away ONLY if that neighbor is in the SAME lane (per the
    // current mode); otherwise it fills with 0, exactly like an
    // independent smaller shifter would. gi+shift is always a generate-
    // time constant, so out-of-range cases are resolved at elaboration
    // (no runtime out-of-bounds array access).
    // The network shifts right (toward bit 0): destination bit gi takes source
    // bit gi+2^k.
    wire [31:0] st0, st1, st2, st3, st4;

    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_st0
            if (gi+16 < 32) begin : ir
                assign st0[gi] = shamt_of[gi][4] ? ((lane_of[gi+16]==lane_of[gi]) ? data_in[gi+16] : 1'b0) : data_in[gi];
            end else begin : oob
                assign st0[gi] = shamt_of[gi][4] ? 1'b0 : data_in[gi];
            end
        end
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_st1
            if (gi+8 < 32) begin : ir
                assign st1[gi] = shamt_of[gi][3] ? ((lane_of[gi+8]==lane_of[gi]) ? st0[gi+8] : 1'b0) : st0[gi];
            end else begin : oob
                assign st1[gi] = shamt_of[gi][3] ? 1'b0 : st0[gi];
            end
        end
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_st2
            if (gi+4 < 32) begin : ir
                assign st2[gi] = shamt_of[gi][2] ? ((lane_of[gi+4]==lane_of[gi]) ? st1[gi+4] : 1'b0) : st1[gi];
            end else begin : oob
                assign st2[gi] = shamt_of[gi][2] ? 1'b0 : st1[gi];
            end
        end
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_st3
            if (gi+2 < 32) begin : ir
                assign st3[gi] = shamt_of[gi][1] ? ((lane_of[gi+2]==lane_of[gi]) ? st2[gi+2] : 1'b0) : st2[gi];
            end else begin : oob
                assign st3[gi] = shamt_of[gi][1] ? 1'b0 : st2[gi];
            end
        end
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_st4
            if (gi+1 < 32) begin : ir
                assign st4[gi] = shamt_of[gi][0] ? ((lane_of[gi+1]==lane_of[gi]) ? st3[gi+1] : 1'b0) : st3[gi];
            end else begin : oob
                assign st4[gi] = shamt_of[gi][0] ? 1'b0 : st3[gi];
            end
        end
    endgenerate

    // A shift amount larger than the format's own width (e.g. a huge real
    // exponent difference whose low 5 bits alias to something small) must
    // still force the result fully to zero -- the 5-stage network above
    // only "sees" exp_diff[4:0], so this catches anything beyond that.
    wire force_zero_0 = (mode_control==FP32) ? (exp_diff_0 > WFP32) :
                         (mode_control==FP16) ? (exp_diff_0 > WFP16) :
                         (mode_control==FP8)  ? (exp_diff_0 > WFP8)  : (exp_diff_0 > WFP4);
    wire force_zero_1 = (mode_control==FP16) ? (exp_diff_1 > WFP16) :
                         (mode_control==FP8)  ? (exp_diff_1 > WFP8)  : (exp_diff_1 > WFP4);
    wire force_zero_2 = (mode_control==FP8)  ? (exp_diff_2 > WFP8)  : (exp_diff_2 > WFP4);
    wire force_zero_3 = (mode_control==FP8)  ? (exp_diff_3 > WFP8)  : (exp_diff_3 > WFP4);
    wire force_zero_4 = (exp_diff_4 > WFP4);
    wire force_zero_5 = (exp_diff_5 > WFP4);
    wire force_zero_6 = (exp_diff_6 > WFP4);
    wire force_zero_7 = (exp_diff_7 > WFP4);

    // Per-lane AND mask: all ones normally, all zeros when that lane's shift
    // amount exceeds its operand width.
    wire [31:0] zero_mask = (mode_control==FP32) ? {32{~force_zero_0}} :
                             (mode_control==FP16) ? {{16{~force_zero_1}}, {16{~force_zero_0}}} :
                             (mode_control==FP8)  ? {{8{~force_zero_3}}, {8{~force_zero_2}},
                                                      {8{~force_zero_1}}, {8{~force_zero_0}}} :
                                                     {{4{~force_zero_7}}, {4{~force_zero_6}},
                                                      {4{~force_zero_5}}, {4{~force_zero_4}},
                                                      {4{~force_zero_3}}, {4{~force_zero_2}},
                                                      {4{~force_zero_1}}, {4{~force_zero_0}}};

    // Shifted operands with over-shifted lanes forced to zero.
    assign data_out = st4 & zero_mask;

endmodule

// =========================================================================
// 7a. SHIFTER PREP (operand ordering and mantissa formatting)
// =========================================================================
// For every lane and format this block
//   - builds each operand's mantissa as {hidden bit, fraction, guard bit};
//     the hidden bit is the exponent-non-zero flag and the guard bit is 0;
//   - decides which operand has the larger magnitude (effective exponent
//     first, then hidden bit + fraction when the exponents are equal);
//   - outputs the larger operand as the base, the smaller as the operand to
//     be aligned (to_shift), the sign of the larger operand as the result
//     sign, and the sign to use if the result is exactly zero;
//   - places the active lanes of the selected format into the nibble-aligned
//     32-bit composite buses base32 / to_shift32.
// The alignment shift itself is applied later by shared_shifter32.
// =========================================================================
module shifter_prep_simd #(
    parameter E   = 8,  parameter M   = 23,  // FP32 (widest) widths
    parameter M16 = 10,
    parameter M8  = 3,
    parameter M4  = 1
)(
    input [1:0] mode_control,

    input [E-1:0] exp_diff_0, exp_diff_1, exp_diff_2, exp_diff_3,
    input [E-1:0] exp_diff_4, exp_diff_5, exp_diff_6, exp_diff_7,
    input signed_a_0, signed_b_0, signed_a_1, signed_b_1,
    input signed_a_2, signed_b_2, signed_a_3, signed_b_3,
    input signed_a_4, signed_b_4, signed_a_5, signed_b_5,
    input signed_a_6, signed_b_6, signed_a_7, signed_b_7,
    input exp_a_notzero_0, exp_b_notzero_0,
    input exp_a_notzero_1, exp_b_notzero_1,
    input exp_a_notzero_2, exp_b_notzero_2,
    input exp_a_notzero_3, exp_b_notzero_3,
    input exp_a_notzero_4, exp_b_notzero_4,
    input exp_a_notzero_5, exp_b_notzero_5,
    input exp_a_notzero_6, exp_b_notzero_6,
    input exp_a_notzero_7, exp_b_notzero_7,
    input exp_a_larger_0, exp_a_larger_1, exp_a_larger_2, exp_a_larger_3,
    input exp_a_larger_4, exp_a_larger_5, exp_a_larger_6, exp_a_larger_7,
    input [M-1:0] frac_a_0, frac_b_0, frac_a_1, frac_b_1,
    input [M-1:0] frac_a_2, frac_b_2, frac_a_3, frac_b_3,
    input [M-1:0] frac_a_4, frac_b_4, frac_a_5, frac_b_5,
    input [M-1:0] frac_a_6, frac_b_6, frac_a_7, frac_b_7,

    // Sign of the larger-magnitude operand in each lane.
    output res_sign_0, res_sign_1, res_sign_2, res_sign_3,
    output res_sign_4, res_sign_5, res_sign_6, res_sign_7,
    // Sign to use IF the result turns out to be exactly zero: IEEE-754 says
    // a same-signed sum (or same-signed zero+zero) keeps that sign (so
    // -0 + -0 = -0), while an opposite-signed exact cancellation rounds to
    // +0. Since signed_xor_i = signed_a_i ^ signed_b_i is 0 exactly when
    // the signs match, "both negative" (signed_a_i & signed_b_i) is exactly
    // that case -- no need to reference signed_xor here at all.
    output zero_sign_0, zero_sign_1, zero_sign_2, zero_sign_3,
    output zero_sign_4, zero_sign_5, zero_sign_6, zero_sign_7,
    output [31:0] base32,      // nibble-aligned "larger magnitude" composite bus
    output [31:0] to_shift32   // nibble-aligned "operand to align" composite bus (not yet shifted)
);

    assign zero_sign_0 = signed_a_0 & signed_b_0;
    assign zero_sign_1 = signed_a_1 & signed_b_1;
    assign zero_sign_2 = signed_a_2 & signed_b_2;
    assign zero_sign_3 = signed_a_3 & signed_b_3;
    assign zero_sign_4 = signed_a_4 & signed_b_4;
    assign zero_sign_5 = signed_a_5 & signed_b_5;
    assign zero_sign_6 = signed_a_6 & signed_b_6;
    assign zero_sign_7 = signed_a_7 & signed_b_7;

    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    localparam WFP32 = M+2;      // 25
    localparam WFP16 = M16+2;    // 12
    localparam WFP8  = M8+2;  // hidden + fraction + guard (5 bits for E4M3)
    localparam WFP4  = M4+2;     // 3

    // ---------------------------------------------------------------
    // FP32 magnitude compare + hidden/guard-bit formatting (lane 0 only)
    // ---------------------------------------------------------------
    // Equal effective exponents -> compare {hidden bit, fraction}; otherwise the
    // exponent comparison decides which operand is larger.
    wire fp32_eq        = (exp_diff_0 == 0);
    wire fp32_a_greater  = fp32_eq ? ({exp_a_notzero_0, frac_a_0} >= {exp_b_notzero_0, frac_b_0}) : exp_a_larger_0;
    // Mantissa = {hidden bit, fraction, guard bit}.
    wire [WFP32-1:0] fp32_mant_a = {exp_a_notzero_0, frac_a_0, 1'b0};
    wire [WFP32-1:0] fp32_mant_b = {exp_b_notzero_0, frac_b_0, 1'b0};
    wire [WFP32-1:0] fp32_base     = fp32_a_greater ? fp32_mant_a : fp32_mant_b;
    wire [WFP32-1:0] fp32_to_shift = fp32_a_greater ? fp32_mant_b : fp32_mant_a;

    // ---------------------------------------------------------------
    // FP16 magnitude compare + hidden/guard-bit formatting (lanes 0,1)
    // ---------------------------------------------------------------
    // Same scheme as FP32, applied per lane with the 10-bit FP16 fraction.
    wire fp16_eq_0       = (exp_diff_0 == 0);
    wire fp16_a_greater_0 = fp16_eq_0 ? ({exp_a_notzero_0, frac_a_0[M16-1:0]} >= {exp_b_notzero_0, frac_b_0[M16-1:0]}) : exp_a_larger_0;
    wire [WFP16-1:0] fp16_mant_a_0 = {exp_a_notzero_0, frac_a_0[M16-1:0], 1'b0};
    wire [WFP16-1:0] fp16_mant_b_0 = {exp_b_notzero_0, frac_b_0[M16-1:0], 1'b0};
    wire [WFP16-1:0] fp16_base_0     = fp16_a_greater_0 ? fp16_mant_a_0 : fp16_mant_b_0;
    wire [WFP16-1:0] fp16_to_shift_0 = fp16_a_greater_0 ? fp16_mant_b_0 : fp16_mant_a_0;

    wire fp16_eq_1       = (exp_diff_1 == 0);
    wire fp16_a_greater_1 = fp16_eq_1 ? ({exp_a_notzero_1, frac_a_1[M16-1:0]} >= {exp_b_notzero_1, frac_b_1[M16-1:0]}) : exp_a_larger_1;
    wire [WFP16-1:0] fp16_mant_a_1 = {exp_a_notzero_1, frac_a_1[M16-1:0], 1'b0};
    wire [WFP16-1:0] fp16_mant_b_1 = {exp_b_notzero_1, frac_b_1[M16-1:0], 1'b0};
    wire [WFP16-1:0] fp16_base_1     = fp16_a_greater_1 ? fp16_mant_a_1 : fp16_mant_b_1;
    wire [WFP16-1:0] fp16_to_shift_1 = fp16_a_greater_1 ? fp16_mant_b_1 : fp16_mant_a_1;

    // ---------------------------------------------------------------
    // FP8 magnitude compare + hidden/guard-bit formatting (lanes 0..3)
    // ---------------------------------------------------------------
    // Same scheme, per lane, with the 3-bit FP8 fraction.
    wire fp8_eq_0        = (exp_diff_0 == 0);
    wire fp8_a_greater_0  = fp8_eq_0 ? ({exp_a_notzero_0, frac_a_0[M8-1:0]} >= {exp_b_notzero_0, frac_b_0[M8-1:0]}) : exp_a_larger_0;
    wire [WFP8-1:0] fp8_mant_a_0 = {exp_a_notzero_0, frac_a_0[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_mant_b_0 = {exp_b_notzero_0, frac_b_0[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_base_0     = fp8_a_greater_0 ? fp8_mant_a_0 : fp8_mant_b_0;
    wire [WFP8-1:0] fp8_to_shift_0 = fp8_a_greater_0 ? fp8_mant_b_0 : fp8_mant_a_0;

    wire fp8_eq_1        = (exp_diff_1 == 0);
    wire fp8_a_greater_1  = fp8_eq_1 ? ({exp_a_notzero_1, frac_a_1[M8-1:0]} >= {exp_b_notzero_1, frac_b_1[M8-1:0]}) : exp_a_larger_1;
    wire [WFP8-1:0] fp8_mant_a_1 = {exp_a_notzero_1, frac_a_1[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_mant_b_1 = {exp_b_notzero_1, frac_b_1[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_base_1     = fp8_a_greater_1 ? fp8_mant_a_1 : fp8_mant_b_1;
    wire [WFP8-1:0] fp8_to_shift_1 = fp8_a_greater_1 ? fp8_mant_b_1 : fp8_mant_a_1;

    wire fp8_eq_2        = (exp_diff_2 == 0);
    wire fp8_a_greater_2  = fp8_eq_2 ? ({exp_a_notzero_2, frac_a_2[M8-1:0]} >= {exp_b_notzero_2, frac_b_2[M8-1:0]}) : exp_a_larger_2;
    wire [WFP8-1:0] fp8_mant_a_2 = {exp_a_notzero_2, frac_a_2[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_mant_b_2 = {exp_b_notzero_2, frac_b_2[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_base_2     = fp8_a_greater_2 ? fp8_mant_a_2 : fp8_mant_b_2;
    wire [WFP8-1:0] fp8_to_shift_2 = fp8_a_greater_2 ? fp8_mant_b_2 : fp8_mant_a_2;

    wire fp8_eq_3        = (exp_diff_3 == 0);
    wire fp8_a_greater_3  = fp8_eq_3 ? ({exp_a_notzero_3, frac_a_3[M8-1:0]} >= {exp_b_notzero_3, frac_b_3[M8-1:0]}) : exp_a_larger_3;
    wire [WFP8-1:0] fp8_mant_a_3 = {exp_a_notzero_3, frac_a_3[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_mant_b_3 = {exp_b_notzero_3, frac_b_3[M8-1:0], 1'b0};
    wire [WFP8-1:0] fp8_base_3     = fp8_a_greater_3 ? fp8_mant_a_3 : fp8_mant_b_3;
    wire [WFP8-1:0] fp8_to_shift_3 = fp8_a_greater_3 ? fp8_mant_b_3 : fp8_mant_a_3;

    // ---------------------------------------------------------------
    // FP4 magnitude compare + hidden/guard-bit formatting (lanes 0..7)
    // ---------------------------------------------------------------
    // Same scheme, per lane, with the 1-bit FP4 fraction.
    wire fp4_eq_0 = (exp_diff_0 == 0);
    wire fp4_a_greater_0 = fp4_eq_0 ? ({exp_a_notzero_0, frac_a_0[M4-1:0]} >= {exp_b_notzero_0, frac_b_0[M4-1:0]}) : exp_a_larger_0;
    wire [WFP4-1:0] fp4_mant_a_0 = {exp_a_notzero_0, frac_a_0[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_0 = {exp_b_notzero_0, frac_b_0[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_0     = fp4_a_greater_0 ? fp4_mant_a_0 : fp4_mant_b_0;
    wire [WFP4-1:0] fp4_to_shift_0 = fp4_a_greater_0 ? fp4_mant_b_0 : fp4_mant_a_0;

    wire fp4_eq_1 = (exp_diff_1 == 0);
    wire fp4_a_greater_1 = fp4_eq_1 ? ({exp_a_notzero_1, frac_a_1[M4-1:0]} >= {exp_b_notzero_1, frac_b_1[M4-1:0]}) : exp_a_larger_1;
    wire [WFP4-1:0] fp4_mant_a_1 = {exp_a_notzero_1, frac_a_1[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_1 = {exp_b_notzero_1, frac_b_1[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_1     = fp4_a_greater_1 ? fp4_mant_a_1 : fp4_mant_b_1;
    wire [WFP4-1:0] fp4_to_shift_1 = fp4_a_greater_1 ? fp4_mant_b_1 : fp4_mant_a_1;

    wire fp4_eq_2 = (exp_diff_2 == 0);
    wire fp4_a_greater_2 = fp4_eq_2 ? ({exp_a_notzero_2, frac_a_2[M4-1:0]} >= {exp_b_notzero_2, frac_b_2[M4-1:0]}) : exp_a_larger_2;
    wire [WFP4-1:0] fp4_mant_a_2 = {exp_a_notzero_2, frac_a_2[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_2 = {exp_b_notzero_2, frac_b_2[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_2     = fp4_a_greater_2 ? fp4_mant_a_2 : fp4_mant_b_2;
    wire [WFP4-1:0] fp4_to_shift_2 = fp4_a_greater_2 ? fp4_mant_b_2 : fp4_mant_a_2;

    wire fp4_eq_3 = (exp_diff_3 == 0);
    wire fp4_a_greater_3 = fp4_eq_3 ? ({exp_a_notzero_3, frac_a_3[M4-1:0]} >= {exp_b_notzero_3, frac_b_3[M4-1:0]}) : exp_a_larger_3;
    wire [WFP4-1:0] fp4_mant_a_3 = {exp_a_notzero_3, frac_a_3[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_3 = {exp_b_notzero_3, frac_b_3[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_3     = fp4_a_greater_3 ? fp4_mant_a_3 : fp4_mant_b_3;
    wire [WFP4-1:0] fp4_to_shift_3 = fp4_a_greater_3 ? fp4_mant_b_3 : fp4_mant_a_3;

    wire fp4_eq_4 = (exp_diff_4 == 0);
    wire fp4_a_greater_4 = fp4_eq_4 ? ({exp_a_notzero_4, frac_a_4[M4-1:0]} >= {exp_b_notzero_4, frac_b_4[M4-1:0]}) : exp_a_larger_4;
    wire [WFP4-1:0] fp4_mant_a_4 = {exp_a_notzero_4, frac_a_4[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_4 = {exp_b_notzero_4, frac_b_4[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_4     = fp4_a_greater_4 ? fp4_mant_a_4 : fp4_mant_b_4;
    wire [WFP4-1:0] fp4_to_shift_4 = fp4_a_greater_4 ? fp4_mant_b_4 : fp4_mant_a_4;

    wire fp4_eq_5 = (exp_diff_5 == 0);
    wire fp4_a_greater_5 = fp4_eq_5 ? ({exp_a_notzero_5, frac_a_5[M4-1:0]} >= {exp_b_notzero_5, frac_b_5[M4-1:0]}) : exp_a_larger_5;
    wire [WFP4-1:0] fp4_mant_a_5 = {exp_a_notzero_5, frac_a_5[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_5 = {exp_b_notzero_5, frac_b_5[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_5     = fp4_a_greater_5 ? fp4_mant_a_5 : fp4_mant_b_5;
    wire [WFP4-1:0] fp4_to_shift_5 = fp4_a_greater_5 ? fp4_mant_b_5 : fp4_mant_a_5;

    wire fp4_eq_6 = (exp_diff_6 == 0);
    wire fp4_a_greater_6 = fp4_eq_6 ? ({exp_a_notzero_6, frac_a_6[M4-1:0]} >= {exp_b_notzero_6, frac_b_6[M4-1:0]}) : exp_a_larger_6;
    wire [WFP4-1:0] fp4_mant_a_6 = {exp_a_notzero_6, frac_a_6[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_6 = {exp_b_notzero_6, frac_b_6[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_6     = fp4_a_greater_6 ? fp4_mant_a_6 : fp4_mant_b_6;
    wire [WFP4-1:0] fp4_to_shift_6 = fp4_a_greater_6 ? fp4_mant_b_6 : fp4_mant_a_6;

    wire fp4_eq_7 = (exp_diff_7 == 0);
    wire fp4_a_greater_7 = fp4_eq_7 ? ({exp_a_notzero_7, frac_a_7[M4-1:0]} >= {exp_b_notzero_7, frac_b_7[M4-1:0]}) : exp_a_larger_7;
    wire [WFP4-1:0] fp4_mant_a_7 = {exp_a_notzero_7, frac_a_7[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_mant_b_7 = {exp_b_notzero_7, frac_b_7[M4-1:0], 1'b0};
    wire [WFP4-1:0] fp4_base_7     = fp4_a_greater_7 ? fp4_mant_a_7 : fp4_mant_b_7;
    wire [WFP4-1:0] fp4_to_shift_7 = fp4_a_greater_7 ? fp4_mant_b_7 : fp4_mant_a_7;

    // ---------------------------------------------------------------
    // OUTPUT MUXING -- align each active lane's data into a single 32-bit
    // composite bus (pure wiring/mux, no arithmetic) so ONE shared 32-bit
    // adder (mux_add_mant_simd) can service every mode. Nibble offsets:
    //   FP32:  lane0 -> bits[24:0]              (nibbles 0-7, chained)
    //   FP16:  lane0 -> bits[11:0]  (nibbles 0-3) lane1 -> bits[27:16] (nibbles 4-7)
    //   FP8:   lane0 -> bits[4:0]   (nib 0-1)  lane1 -> bits[12:8]  (nib 2-3)
    //          lane2 -> bits[20:16] (nib 4-5)  lane3 -> bits[28:24] (nib 6-7)
    //          (5-bit operands; each lane's sum, with overflow bit, spans 6 bits)
    //   FP4:   lane i -> bits[4i+2:4i]  (one nibble each, i=0..7)
    // ---------------------------------------------------------------
    // Per-lane "operand a has the larger magnitude" flag for the active format.
    // Lanes that do not exist in the current mode read 0.
    wire mant_a_greater_0 = (mode_control == FP4) ? fp4_a_greater_0 :
                             (mode_control == FP8) ? fp8_a_greater_0 :
                             (mode_control == FP16) ? fp16_a_greater_0 : fp32_a_greater;
    wire mant_a_greater_1 = (mode_control == FP4) ? fp4_a_greater_1 :
                             (mode_control == FP8) ? fp8_a_greater_1 :
                             (mode_control == FP16) ? fp16_a_greater_1 : 1'b0;
    wire mant_a_greater_2 = (mode_control == FP4) ? fp4_a_greater_2 :
                             (mode_control == FP8) ? fp8_a_greater_2 : 1'b0;
    wire mant_a_greater_3 = (mode_control == FP4) ? fp4_a_greater_3 :
                             (mode_control == FP8) ? fp8_a_greater_3 : 1'b0;
    wire mant_a_greater_4 = (mode_control == FP4) ? fp4_a_greater_4 : 1'b0;
    wire mant_a_greater_5 = (mode_control == FP4) ? fp4_a_greater_5 : 1'b0;
    wire mant_a_greater_6 = (mode_control == FP4) ? fp4_a_greater_6 : 1'b0;
    wire mant_a_greater_7 = (mode_control == FP4) ? fp4_a_greater_7 : 1'b0;

    // Result sign = sign of the larger-magnitude operand; inactive lanes give 0.
    assign res_sign_0 = mant_a_greater_0 ? signed_a_0 : signed_b_0;
    assign res_sign_1 = (mode_control != FP32) ? (mant_a_greater_1 ? signed_a_1 : signed_b_1) : 1'b0;
    assign res_sign_2 = (mode_control == FP8 || mode_control == FP4) ? (mant_a_greater_2 ? signed_a_2 : signed_b_2) : 1'b0;
    assign res_sign_3 = (mode_control == FP8 || mode_control == FP4) ? (mant_a_greater_3 ? signed_a_3 : signed_b_3) : 1'b0;
    assign res_sign_4 = (mode_control == FP4) ? (mant_a_greater_4 ? signed_a_4 : signed_b_4) : 1'b0;
    assign res_sign_5 = (mode_control == FP4) ? (mant_a_greater_5 ? signed_a_5 : signed_b_5) : 1'b0;
    assign res_sign_6 = (mode_control == FP4) ? (mant_a_greater_6 ? signed_a_6 : signed_b_6) : 1'b0;
    assign res_sign_7 = (mode_control == FP4) ? (mant_a_greater_7 ? signed_a_7 : signed_b_7) : 1'b0;

    // Composite "larger magnitude" bus: the operand to which the aligned
    // operand is added or from which it is subtracted.
    assign base32    = (mode_control == FP32) ? {7'b0, fp32_base} :
                             (mode_control == FP16) ? {4'b0, fp16_base_1, 4'b0, fp16_base_0} :
                             (mode_control == FP8)  ? {{(8-WFP8){1'b0}}, fp8_base_3, {(8-WFP8){1'b0}}, fp8_base_2,
                                                        {(8-WFP8){1'b0}}, fp8_base_1, {(8-WFP8){1'b0}}, fp8_base_0} :
                                                       {1'b0, fp4_base_7, 1'b0, fp4_base_6,
                                                        1'b0, fp4_base_5, 1'b0, fp4_base_4,
                                                        1'b0, fp4_base_3, 1'b0, fp4_base_2,
                                                        1'b0, fp4_base_1, 1'b0, fp4_base_0};

    // Composite "operand to align" bus -- NOT yet shifted. The shift itself
    // happens once, below, in the shared segmented barrel shifter.
    assign to_shift32 = (mode_control == FP32) ? {7'b0, fp32_to_shift} :
                              (mode_control == FP16) ? {4'b0, fp16_to_shift_1, 4'b0, fp16_to_shift_0} :
                              (mode_control == FP8)  ? {{(8-WFP8){1'b0}}, fp8_to_shift_3, {(8-WFP8){1'b0}}, fp8_to_shift_2,
                                                         {(8-WFP8){1'b0}}, fp8_to_shift_1, {(8-WFP8){1'b0}}, fp8_to_shift_0} :
                                                        {1'b0, fp4_to_shift_7, 1'b0, fp4_to_shift_6,
                                                         1'b0, fp4_to_shift_5, 1'b0, fp4_to_shift_4,
                                                         1'b0, fp4_to_shift_3, 1'b0, fp4_to_shift_2,
                                                         1'b0, fp4_to_shift_1, 1'b0, fp4_to_shift_0};


endmodule



// =========================================================================
// 7b. SHIFT + ADD (alignment shift and mantissa add in one block)
// =========================================================================
// Combines shared_shifter32 and mux_add_mant_simd and slices the shared sum
// back into per-lane results. The pipelined top level does not instantiate
// this module; it places a register between the shifter and the adder and
// uses shared_shifter32 and add_slice_simd directly.
// =========================================================================
module shift_add_simd #(
    parameter E  = 8,
    parameter M  = 23,
    parameter M16 = 10,
    parameter M8  = 3,
    parameter M4  = 1
)(
    input [1:0] mode_control,
    input [31:0] base32, to_shift32,      // from shifter_prep_simd (registered)
    input [E-1:0] exp_diff_0, exp_diff_1, exp_diff_2, exp_diff_3,
    input [E-1:0] exp_diff_4, exp_diff_5, exp_diff_6, exp_diff_7,
    input signed_xor_0, signed_xor_1, signed_xor_2, signed_xor_3,
    input signed_xor_4, signed_xor_5, signed_xor_6, signed_xor_7,

    output [M+2:0] frac_out_0, frac_out_1, frac_out_2, frac_out_3, // 26 bits: sum incl. overflow bit
    output [M+2:0] frac_out_4, frac_out_5, frac_out_6, frac_out_7
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    localparam WFP32 = M+2;
    localparam WFP16 = M16+2;
    localparam WFP8  = M8+2;  // hidden + fraction + guard (5 bits for E4M3)
    localparam WFP4  = M4+2;

    wire [31:0] shifted32;
    wire [31:0] sum32;

    // ---------------------------------------------------------------
    // ONE SHARED SEGMENTED BARREL SHIFTER (replaces 4 dedicated shifters:
    // a 25-bit one for FP32, two 12-bit ones for FP16, four 5-bit ones for
    // FP8, eight 3-bit ones for FP4) -- same physical 32-bit shift network
    // reused by every mode.
    // ---------------------------------------------------------------
    shared_shifter32 #( .E(E), .M8(M8) ) u_shared_shifter (
        .mode_control(mode_control),
        .data_in(to_shift32),
        .exp_diff_0(exp_diff_0), .exp_diff_1(exp_diff_1),
        .exp_diff_2(exp_diff_2), .exp_diff_3(exp_diff_3),
        .exp_diff_4(exp_diff_4), .exp_diff_5(exp_diff_5),
        .exp_diff_6(exp_diff_6), .exp_diff_7(exp_diff_7),
        .data_out(shifted32)
    );

    // ---------------------------------------------------------------
    // ONE SHARED 32-BIT ADDER
    // ---------------------------------------------------------------
    mux_add_mant_simd u_mux_add_mant_simd (
        .mode_control(mode_control),
        .signed_xor_0(signed_xor_0), .signed_xor_1(signed_xor_1),
        .signed_xor_2(signed_xor_2), .signed_xor_3(signed_xor_3),
        .signed_xor_4(signed_xor_4), .signed_xor_5(signed_xor_5),
        .signed_xor_6(signed_xor_6), .signed_xor_7(signed_xor_7),
        .base32(base32), .shifted32(shifted32),
        .sum32(sum32)
    );

    // Slice the shared sum back out per lane (wiring only, zero-extended into
    // the common 26-bit frac_out bus that pack_simd expects).
    // frac_out_k = lane k's sum including its overflow bit, right-aligned and
    // zero-extended to M+3 = 26 bits. Lane k sits at bit offset 16k (FP16),
    // 8k (FP8) or 4k (FP4); FP32 uses lane 0 only.
    assign frac_out_0 = (mode_control == FP32) ? sum32[WFP32:0] :
                         (mode_control == FP16) ? {{(M+3-WFP16-1){1'b0}}, sum32[WFP16:0]} :
                         (mode_control == FP8)  ? {{(M+3-WFP8-1){1'b0}}, sum32[WFP8:0]} :
                                                   {{(M+3-WFP4-1){1'b0}}, sum32[WFP4:0]};

    assign frac_out_1 = (mode_control == FP16) ? {{(M+3-WFP16-1){1'b0}}, sum32[16+WFP16:16]} :
                         (mode_control == FP8)  ? {{(M+3-WFP8-1){1'b0}}, sum32[8+WFP8:8]} :
                         (mode_control == FP4)  ? {{(M+3-WFP4-1){1'b0}}, sum32[4+WFP4:4]} :
                                                   {(M+3){1'b0}};

    assign frac_out_2 = (mode_control == FP8) ? {{(M+3-WFP8-1){1'b0}}, sum32[16+WFP8:16]} :
                         (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[8+WFP4:8]} :
                                                  {(M+3){1'b0}};

    assign frac_out_3 = (mode_control == FP8) ? {{(M+3-WFP8-1){1'b0}}, sum32[24+WFP8:24]} :
                         (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[12+WFP4:12]} :
                                                  {(M+3){1'b0}};

    assign frac_out_4 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[16+WFP4:16]} : {(M+3){1'b0}};
    assign frac_out_5 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[20+WFP4:20]} : {(M+3){1'b0}};
    assign frac_out_6 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[24+WFP4:24]} : {(M+3){1'b0}};
    assign frac_out_7 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[28+WFP4:28]} : {(M+3){1'b0}};

endmodule


// =========================================================================
// 7c. ADD + SLICE (mantissa add stage used by the pipelined top level)
// =========================================================================
// Takes the base operand and the already-aligned operand (shifted32, from
// the preceding register stage), adds or subtracts them in the shared
// segmented adder (mux_add_mant_simd) and slices the 32-bit sum into
// per-lane results.
// =========================================================================
module add_slice_simd #(
    parameter E  = 8,
    parameter M  = 23,
    parameter M16 = 10,
    parameter M8  = 3,
    parameter M4  = 1
)(
    input [1:0] mode_control,
    input [31:0] base32, shifted32,       // shifted32 comes from the shifter stage (registered)
    input signed_xor_0, signed_xor_1, signed_xor_2, signed_xor_3,
    input signed_xor_4, signed_xor_5, signed_xor_6, signed_xor_7,
    output [M+2:0] frac_out_0, frac_out_1, frac_out_2, frac_out_3,
    output [M+2:0] frac_out_4, frac_out_5, frac_out_6, frac_out_7
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    localparam WFP32 = M+2;
    localparam WFP16 = M16+2;
    localparam WFP8  = M8+2;  // hidden + fraction + guard (5 bits for E4M3)
    localparam WFP4  = M4+2;
    wire [31:0] sum32;

    mux_add_mant_simd u_mux_add_mant_simd (
        .mode_control(mode_control),
        .signed_xor_0(signed_xor_0), .signed_xor_1(signed_xor_1),
        .signed_xor_2(signed_xor_2), .signed_xor_3(signed_xor_3),
        .signed_xor_4(signed_xor_4), .signed_xor_5(signed_xor_5),
        .signed_xor_6(signed_xor_6), .signed_xor_7(signed_xor_7),
        .base32(base32), .shifted32(shifted32),
        .sum32(sum32)
    );

    // Slice the shared sum back out per lane (wiring only, zero-extended into
    // the common 26-bit frac_out bus that pack_simd expects).
    // frac_out_k = lane k's sum including its overflow bit, right-aligned and
    // zero-extended to M+3 = 26 bits. Lane k sits at bit offset 16k (FP16),
    // 8k (FP8) or 4k (FP4); FP32 uses lane 0 only.
    assign frac_out_0 = (mode_control == FP32) ? sum32[WFP32:0] :
                         (mode_control == FP16) ? {{(M+3-WFP16-1){1'b0}}, sum32[WFP16:0]} :
                         (mode_control == FP8)  ? {{(M+3-WFP8-1){1'b0}}, sum32[WFP8:0]} :
                                                   {{(M+3-WFP4-1){1'b0}}, sum32[WFP4:0]};

    assign frac_out_1 = (mode_control == FP16) ? {{(M+3-WFP16-1){1'b0}}, sum32[16+WFP16:16]} :
                         (mode_control == FP8)  ? {{(M+3-WFP8-1){1'b0}}, sum32[8+WFP8:8]} :
                         (mode_control == FP4)  ? {{(M+3-WFP4-1){1'b0}}, sum32[4+WFP4:4]} :
                                                   {(M+3){1'b0}};

    assign frac_out_2 = (mode_control == FP8) ? {{(M+3-WFP8-1){1'b0}}, sum32[16+WFP8:16]} :
                         (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[8+WFP4:8]} :
                                                  {(M+3){1'b0}};

    assign frac_out_3 = (mode_control == FP8) ? {{(M+3-WFP8-1){1'b0}}, sum32[24+WFP8:24]} :
                         (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[12+WFP4:12]} :
                                                  {(M+3){1'b0}};

    assign frac_out_4 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[16+WFP4:16]} : {(M+3){1'b0}};
    assign frac_out_5 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[20+WFP4:20]} : {(M+3){1'b0}};
    assign frac_out_6 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[24+WFP4:24]} : {(M+3){1'b0}};
    assign frac_out_7 = (mode_control == FP4) ? {{(M+3-WFP4-1){1'b0}}, sum32[28+WFP4:28]} : {(M+3){1'b0}};

endmodule




// =========================================================================
// 8. PACK (normalize + reassemble, 4 formats x 8 lanes)
//    Split into two modules around a pipeline register:
//      pack_decide_simd  - zero/overflow detection, leading-zero count,
//                          shift amount and final exponent
//      pack_execute_simd - normalizing left shift, fraction extraction and
//                          encoding of each format's result word
//    There is no separate rounding step; fractions are truncated.
// =========================================================================
// =========================================================================
// 8a. PACK DECIDE -- leading-zero-count + shift-amount + final-exponent
//     decision, split out of pack_simd as its own pipeline stage. This is
//     everything EXCEPT the actual barrel-shift-left and final bit
//     encoding, which move to pack_execute_simd on the other side of a
//     pipeline register -- splitting the normalizer's two deepest pieces
//     (LZC and the shift) into separate cycles for a shorter critical path.
//
// AREA/OUTPUT OPTIMIZATION: rather than registering both pre_align and
// ovf_align (and a separate ovf flag) across the pipeline boundary, note
// that the overflow case never needs further shifting -- its mantissa is
// already exactly right, so it is folded into the SAME "shift by act_shift"
// mechanism used by the normal case, just with act_shift forced to 0:
//   shift_src = ovf ? ovf_align : pre_align
//   act_shift = ovf ? 0 : (clamp ? max_shift : lzc)
// pack_execute_simd then only needs one shift (shift_src << act_shift),
// covering both the overflow and normal cases identically, and a separate
// is_zero flag for the (also shift-free) zero case.
// =========================================================================
module pack_decide_simd #(
    parameter FTZ_EN = 1,   // 1: flush subnormal FP32/FP16 results to signed zero
    parameter E32 = 8,  parameter M32 = 23,  // FP32: 8-bit exponent, 23-bit fraction
    parameter E16 = 5,  parameter M16 = 10,  // FP16: 5-bit exponent, 10-bit fraction
    parameter E8  = 4,  parameter M8  = 3,  // FP8 (E4M3): 4-bit exponent, 3-bit fraction
    parameter E4  = 2,  parameter M4  = 1  // FP4 (E2M1): 2-bit exponent, 1-bit fraction
)(
    input clk, input ce,
    input [1:0] mode_control,

    // mant_res_k: lane k's sum from the adder stage, including the overflow bit.
    input [M32+2:0] mant_res_0, mant_res_1, mant_res_2, mant_res_3, // 26 bits each
    input [M32+2:0] mant_res_4, mant_res_5, mant_res_6, mant_res_7,
    // exp_common_k: larger effective exponent of lane k.
    input [E32-1:0] exp_common_0, exp_common_1, exp_common_2, exp_common_3,
    input [E32-1:0] exp_common_4, exp_common_5, exp_common_6, exp_common_7,

    // Per-lane decisions handed to pack_execute_simd:
    //   is_zero   : the result is exactly zero (or flushed to zero)
    //   shift_src : mantissa to normalize (overflow-corrected sum on overflow,
    //               otherwise the left-justified sum)
    //   act_shift : left-shift amount that normalizes shift_src
    //   final_exp : result exponent (one extra bit so overflow can be detected)
    output is_zero_0, output [M32+1:0] shift_src_0, output [4:0] act_shift_0, output [E32:0] final_exp_0,
    output is_zero_1, output [M16+1:0] shift_src_1, output [3:0] act_shift_1, output [E32:0] final_exp_1,
    output is_zero_2, output [M8+1:0]  shift_src_2, output [2:0] act_shift_2, output [E32:0] final_exp_2,
    output is_zero_3, output [M8+1:0]  shift_src_3, output [2:0] act_shift_3, output [E32:0] final_exp_3,
    output is_zero_4, output [M4+1:0]  shift_src_4, output [1:0] act_shift_4, output [E32:0] final_exp_4,
    output is_zero_5, output [M4+1:0]  shift_src_5, output [1:0] act_shift_5, output [E32:0] final_exp_5,
    output is_zero_6, output [M4+1:0]  shift_src_6, output [1:0] act_shift_6, output [E32:0] final_exp_6,
    output is_zero_7, output [M4+1:0]  shift_src_7, output [1:0] act_shift_7, output [E32:0] final_exp_7,

    // Flush-to-zero flags (subnormal result), reusing the LZC clamp compare.
    // ftz_0: FP32 or FP16-lo result is subnormal.  ftz_1: FP16-hi result is subnormal.
    output ftz_0, output ftz_1
);

    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    localparam WFP32 = M32+2; // 25
    localparam WFP16 = M16+2; // 12
    localparam WFP8  = M8+2;  // hidden + fraction + guard (5 bits for E4M3)
    localparam WFP4  = M4+2;  // 3

    // ---- generic leading-zero counters (priority encoders). Each examines the
    // full left-justified W-bit field and returns the number of leading zeros,
    // saturating at W-1 for an all-zero input. ----
    function [4:0] lzc25(input [WFP32-1:0] v);
        integer k;
        begin
            lzc25 = WFP32-1;
            for (k = WFP32-1; k >= 0; k = k - 1)
                if (v[k] && (lzc25 == WFP32-1)) lzc25 = (WFP32-1) - k;
        end
    endfunction

    function [3:0] lzc12(input [WFP16-1:0] v);
        integer k;
        begin
            lzc12 = WFP16-1;
            for (k = WFP16-1; k >= 0; k = k - 1)
                if (v[k] && (lzc12 == WFP16-1)) lzc12 = (WFP16-1) - k;
        end
    endfunction

    function [2:0] lzc6(input [WFP8-1:0] v);
        integer k;
        begin
            lzc6 = WFP8-1;
            for (k = WFP8-1; k >= 0; k = k - 1)
                if (v[k] && (lzc6 == WFP8-1)) lzc6 = (WFP8-1) - k;
        end
    endfunction

    // =====================================================================
    // LANE 0 -- shared 25-bit LZC across all four formats
    // =====================================================================
    // ovf0: carry-out of the lane sum (bit W, just above the hidden-bit position).
    wire ovf0 = (mode_control==FP32) ? mant_res_0[WFP32] :
                (mode_control==FP16) ? mant_res_0[WFP16] :
                (mode_control==FP8)  ? mant_res_0[WFP8]  : mant_res_0[WFP4];

    // pre_align0: lane sum without the overflow bit, left-justified into 25 bits
    // so a single leading-zero counter serves every format.
    wire [WFP32-1:0] pre_align0 = (mode_control==FP32) ? mant_res_0[WFP32-1:0] :
                                   (mode_control==FP16) ? {mant_res_0[WFP16-1:0], {(WFP32-WFP16){1'b0}}} :
                                   (mode_control==FP8)  ? {mant_res_0[WFP8-1:0],  {(WFP32-WFP8){1'b0}}} :
                                                           {mant_res_0[WFP4-1:0],  {(WFP32-WFP4){1'b0}}};

    // ovf_align0: the sum right-shifted by one (the overflow bit becomes the
    // new MSB), left-justified the same way.
    wire [WFP32-1:0] ovf_align0 = (mode_control==FP32) ? mant_res_0[WFP32:1] :
                                   (mode_control==FP16) ? {mant_res_0[WFP16:1], {(WFP32-WFP16){1'b0}}} :
                                   (mode_control==FP8)  ? {mant_res_0[WFP8:1],  {(WFP32-WFP8){1'b0}}} :
                                                           {mant_res_0[WFP4:1],  {(WFP32-WFP4){1'b0}}};

    // The result is exactly zero when the sum is zero and there is no overflow.
    wire iz0_d = (pre_align0 == 0) && !ovf0;
    wire [WFP32-1:0] ss0_d = ovf0 ? ovf_align0 : pre_align0;

    // ---- Stage 3a: LZC + exp-1 (registered), Stage 3b: clamp / exponent / FTZ ----
    wire [4:0]   lzc0_d = lzc25(pre_align0);
    wire [E32:0] ms0_d  = (exp_common_0 > 0) ? (exp_common_0 - 1'b1) : {(E32+1){1'b0}};
    reg ovf0_q, iz0_q; reg [4:0] lzc0_q; reg [E32:0] ms0_q; reg [E32-1:0] ec0_q;
    reg [WFP32-1:0] ss0_q; reg [1:0] mode0_q;
    always @(posedge clk) if (ce) begin
        ovf0_q <= ovf0; iz0_q <= iz0_d; lzc0_q <= lzc0_d; ms0_q <= ms0_d;
        ec0_q <= exp_common_0; ss0_q <= ss0_d; mode0_q <= mode_control;
    end
    // Clamp: normalizing by the full leading-zero count would push the exponent
    // below 1, so the shift is limited to (exp - 1) and the result is subnormal.
    wire         clamp0 = !ovf0_q && !iz0_q && ({4'b0, lzc0_q} > ms0_q);
    wire [4:0]   norm_shift0 = clamp0 ? ms0_q[4:0] : lzc0_q;
    wire [E32:0] ec0x = {1'b0, ec0_q};
    wire [E32:0] ec0_p1 = ec0x + 1'b1;
    // Exponent after normalization (exp - leading zeros); a zero exponent is
    // bumped to 1.
    wire [E32:0] ec0_sub = (ec0_q == 0) ? ec0_p1 : (ec0x - {4'b0, lzc0_q});   // runs parallel to the compare
    assign is_zero_0   = iz0_q;
    assign shift_src_0 = ss0_q;
    // Overflow: exp + 1. Zero or clamped (subnormal): 0. Otherwise: exp - lzc.
    assign final_exp_0 = ovf0_q ? ec0_p1 : (iz0_q | clamp0) ? {(E32+1){1'b0}} : ec0_sub;
    // clamp0 is only 1 for a non-zero, non-overflow result whose normalization would
    // drive the exponent below 1 -> exactly the subnormal-result condition.
    assign ftz_0 = (FTZ_EN != 0) && clamp0 && (mode0_q == FP32 || mode0_q == FP16);
    // No left shift is applied for overflow or zero results.
    assign act_shift_0 = (ovf0_q | iz0_q) ? 5'd0 : norm_shift0;

    // =====================================================================
    // LANE 1 -- shared 12-bit LZC across FP16/FP8/FP4
    // =====================================================================
    // Same decision logic as lane 0, sized for the 12-bit FP16 field (also used
    // by FP8 / FP4 lane 1).
    wire ovf1 = (mode_control==FP16) ? mant_res_1[WFP16] :
                (mode_control==FP8)  ? mant_res_1[WFP8]  : mant_res_1[WFP4];

    wire [WFP16-1:0] pre_align1 = (mode_control==FP16) ? mant_res_1[WFP16-1:0] :
                                   (mode_control==FP8)  ? {mant_res_1[WFP8-1:0], {(WFP16-WFP8){1'b0}}} :
                                                           {mant_res_1[WFP4-1:0], {(WFP16-WFP4){1'b0}}};

    wire [WFP16-1:0] ovf_align1 = (mode_control==FP16) ? mant_res_1[WFP16:1] :
                                   (mode_control==FP8)  ? {mant_res_1[WFP8:1], {(WFP16-WFP8){1'b0}}} :
                                                           {mant_res_1[WFP4:1], {(WFP16-WFP4){1'b0}}};

    wire iz1_d = (pre_align1 == 0) && !ovf1;
    wire [WFP16-1:0] ss1_d = ovf1 ? ovf_align1 : pre_align1;

    wire [3:0]   lzc1_d = lzc12(pre_align1);
    wire [E32:0] ms1_d  = (exp_common_1 > 0) ? (exp_common_1 - 1'b1) : {(E32+1){1'b0}};
    reg ovf1_q, iz1_q; reg [3:0] lzc1_q; reg [E32:0] ms1_q; reg [E32-1:0] ec1_q;
    reg [WFP16-1:0] ss1_q; reg [1:0] mode1_q;
    always @(posedge clk) if (ce) begin
        ovf1_q <= ovf1; iz1_q <= iz1_d; lzc1_q <= lzc1_d; ms1_q <= ms1_d;
        ec1_q <= exp_common_1; ss1_q <= ss1_d; mode1_q <= mode_control;
    end
    wire         clamp1 = !ovf1_q && !iz1_q && ({5'b0, lzc1_q} > ms1_q);
    wire [3:0]   norm_shift1 = clamp1 ? ms1_q[3:0] : lzc1_q;
    wire [E32:0] ec1x = {1'b0, ec1_q};
    wire [E32:0] ec1_p1 = ec1x + 1'b1;
    wire [E32:0] ec1_sub = (ec1_q == 0) ? ec1_p1 : (ec1x - {5'b0, lzc1_q});
    assign is_zero_1   = iz1_q;
    assign shift_src_1 = ss1_q;
    assign final_exp_1 = ovf1_q ? ec1_p1 : (iz1_q | clamp1) ? {(E32+1){1'b0}} : ec1_sub;
    // Only FP16 has a second flushable lane (FP32 uses lane 0 only).
    assign ftz_1 = (FTZ_EN != 0) && clamp1 && (mode1_q == FP16);
    assign act_shift_1 = (ovf1_q | iz1_q) ? 4'd0 : norm_shift1;

    // =====================================================================
    // ---- lanes 2..7: whole decision fits in stage 3a; register the outputs (aligned with lanes 0/1 stage 3b)
    // d_*: combinational decision for lanes 2..7, computed straight from the
    // adder-stage result; the q_* registers below hold the registered copies.
    wire d_is_zero_2; wire [M8+1:0] d_shift_src_2; wire [3-1:0] d_act_shift_2; wire [E32:0] d_final_exp_2;
    wire d_is_zero_3; wire [M8+1:0] d_shift_src_3; wire [3-1:0] d_act_shift_3; wire [E32:0] d_final_exp_3;
    wire d_is_zero_4; wire [M4+1:0] d_shift_src_4; wire [2-1:0] d_act_shift_4; wire [E32:0] d_final_exp_4;
    wire d_is_zero_5; wire [M4+1:0] d_shift_src_5; wire [2-1:0] d_act_shift_5; wire [E32:0] d_final_exp_5;
    wire d_is_zero_6; wire [M4+1:0] d_shift_src_6; wire [2-1:0] d_act_shift_6; wire [E32:0] d_final_exp_6;
    wire d_is_zero_7; wire [M4+1:0] d_shift_src_7; wire [2-1:0] d_act_shift_7; wire [E32:0] d_final_exp_7;

    // LANES 2 & 3 -- shared 5-bit LZC across FP8/FP4
    // =====================================================================
    // DECIDE_8_4(idx): decision logic for one FP8/FP4 lane (lanes 2 and 3).
    // Same steps as lane 0: overflow / zero detect, leading-zero count on the
    // left-justified sum, clamp against (exp - 1), final exponent and
    // normalization shift amount.
    `define DECIDE_8_4(idx) \
        wire ovf_``idx`` = (mode_control==FP8) ? mant_res_``idx``[WFP8] : mant_res_``idx``[WFP4]; \
        wire [WFP8-1:0] pre_align_``idx`` = (mode_control==FP8) ? mant_res_``idx``[WFP8-1:0] : \
                                             {mant_res_``idx``[WFP4-1:0], {(WFP8-WFP4){1'b0}}}; \
        wire [WFP8-1:0] ovf_align_``idx`` = (mode_control==FP8) ? mant_res_``idx``[WFP8:1] : \
                                             {mant_res_``idx``[WFP4:1], {(WFP8-WFP4){1'b0}}}; \
        assign d_is_zero_``idx`` = (pre_align_``idx`` == 0) && !ovf_``idx``; \
        assign d_shift_src_``idx`` = ovf_``idx`` ? ovf_align_``idx`` : pre_align_``idx``; \
        reg [2:0] lzc_``idx``; \
        reg [E32:0] max_shift_``idx``; \
        reg clamp_``idx``; \
        reg [2:0] norm_shift_``idx``; \
        reg [E32:0] fexp_``idx``; \
        always @(*) begin \
            if (ovf_``idx``) begin \
                fexp_``idx`` = exp_common_``idx`` + 1'b1; \
            end else if (d_is_zero_``idx``) begin \
                fexp_``idx`` = {(E32+1){1'b0}}; \
            end else begin \
                lzc_``idx``       = lzc6(pre_align_``idx``); \
                max_shift_``idx`` = (exp_common_``idx`` > 0) ? (exp_common_``idx`` - 1'b1) : {(E32+1){1'b0}}; \
                clamp_``idx``     = ({2'b00, lzc_``idx``} > max_shift_``idx``); \
                norm_shift_``idx`` = clamp_``idx`` ? max_shift_``idx``[2:0] : lzc_``idx``; \
                fexp_``idx`` = clamp_``idx`` ? {(E32+1){1'b0}} : \
                                (exp_common_``idx`` == 0) ? (exp_common_``idx`` + 1'b1) : \
                                                             (exp_common_``idx`` - norm_shift_``idx``); \
            end \
        end \
        assign d_final_exp_``idx`` = fexp_``idx``; \
        assign d_act_shift_``idx`` = ovf_``idx`` ? 3'd0 : (d_is_zero_``idx`` ? 3'd0 : norm_shift_``idx``);

    `DECIDE_8_4(2)
    `DECIDE_8_4(3)
    `undef DECIDE_8_4

    // =====================================================================
    // LANES 4-7 -- FP4-only, dedicated small (3-bit) decision logic
    // =====================================================================
    // DECIDE_4_ONLY(idx): FP4-only lanes (4..7). The 3-bit sum needs just a
    // 2-bit leading-zero count.
    `define DECIDE_4_ONLY(idx) \
        wire ovf4_``idx`` = mant_res_``idx``[WFP4]; \
        wire [WFP4-1:0] pre4_``idx`` = mant_res_``idx``[WFP4-1:0]; \
        assign d_is_zero_``idx`` = (pre4_``idx`` == 0) && !ovf4_``idx``; \
        assign d_shift_src_``idx`` = ovf4_``idx`` ? mant_res_``idx``[WFP4:1] : pre4_``idx``; \
        reg [1:0] lzc4_``idx``; \
        reg [E32:0] max_shift4_``idx``; \
        reg clamp4_``idx``; \
        reg [1:0] norm_shift4_``idx``; \
        reg [E32:0] fexp4_``idx``; \
        always @(*) begin \
            if (ovf4_``idx``) begin \
                fexp4_``idx`` = exp_common_``idx`` + 1'b1; \
            end else if (d_is_zero_``idx``) begin \
                fexp4_``idx`` = {(E32+1){1'b0}}; \
            end else begin \
                lzc4_``idx``       = pre4_``idx``[2] ? 2'd0 : pre4_``idx``[1] ? 2'd1 : 2'd2; \
                max_shift4_``idx`` = (exp_common_``idx`` > 0) ? (exp_common_``idx`` - 1'b1) : {(E32+1){1'b0}}; \
                clamp4_``idx``     = ({3'b0, lzc4_``idx``} > max_shift4_``idx``); \
                norm_shift4_``idx`` = clamp4_``idx`` ? max_shift4_``idx``[1:0] : lzc4_``idx``; \
                fexp4_``idx`` = clamp4_``idx`` ? {(E32+1){1'b0}} : \
                                 (exp_common_``idx`` == 0) ? (exp_common_``idx`` + 1'b1) : \
                                                              (exp_common_``idx`` - norm_shift4_``idx``); \
            end \
        end \
        assign d_final_exp_``idx`` = fexp4_``idx``; \
        assign d_act_shift_``idx`` = ovf4_``idx`` ? 2'd0 : (d_is_zero_``idx`` ? 2'd0 : norm_shift4_``idx``);

    `DECIDE_4_ONLY(4)
    `DECIDE_4_ONLY(5)
    `DECIDE_4_ONLY(6)
    `DECIDE_4_ONLY(7)
    `undef DECIDE_4_ONLY


    // Register the lane 2..7 decisions so they line up with the lane 0/1
    // results, which are registered inside this module.
    reg q_iz_2,q_iz_3,q_iz_4,q_iz_5,q_iz_6,q_iz_7;
    reg [M8+1:0] q_ss_2,q_ss_3; reg [M4+1:0] q_ss_4,q_ss_5,q_ss_6,q_ss_7;
    reg [2:0] q_as_2,q_as_3; reg [1:0] q_as_4,q_as_5,q_as_6,q_as_7;
    reg [E32:0] q_fe_2,q_fe_3,q_fe_4,q_fe_5,q_fe_6,q_fe_7;
    always @(posedge clk) if (ce) begin
        q_iz_2 <= d_is_zero_2; q_ss_2 <= d_shift_src_2; q_as_2 <= d_act_shift_2; q_fe_2 <= d_final_exp_2;
        q_iz_3 <= d_is_zero_3; q_ss_3 <= d_shift_src_3; q_as_3 <= d_act_shift_3; q_fe_3 <= d_final_exp_3;
        q_iz_4 <= d_is_zero_4; q_ss_4 <= d_shift_src_4; q_as_4 <= d_act_shift_4; q_fe_4 <= d_final_exp_4;
        q_iz_5 <= d_is_zero_5; q_ss_5 <= d_shift_src_5; q_as_5 <= d_act_shift_5; q_fe_5 <= d_final_exp_5;
        q_iz_6 <= d_is_zero_6; q_ss_6 <= d_shift_src_6; q_as_6 <= d_act_shift_6; q_fe_6 <= d_final_exp_6;
        q_iz_7 <= d_is_zero_7; q_ss_7 <= d_shift_src_7; q_as_7 <= d_act_shift_7; q_fe_7 <= d_final_exp_7;
    end
    assign is_zero_2 = q_iz_2; assign shift_src_2 = q_ss_2; assign act_shift_2 = q_as_2; assign final_exp_2 = q_fe_2;
    assign is_zero_3 = q_iz_3; assign shift_src_3 = q_ss_3; assign act_shift_3 = q_as_3; assign final_exp_3 = q_fe_3;
    assign is_zero_4 = q_iz_4; assign shift_src_4 = q_ss_4; assign act_shift_4 = q_as_4; assign final_exp_4 = q_fe_4;
    assign is_zero_5 = q_iz_5; assign shift_src_5 = q_ss_5; assign act_shift_5 = q_as_5; assign final_exp_5 = q_fe_5;
    assign is_zero_6 = q_iz_6; assign shift_src_6 = q_ss_6; assign act_shift_6 = q_as_6; assign final_exp_6 = q_fe_6;
    assign is_zero_7 = q_iz_7; assign shift_src_7 = q_ss_7; assign act_shift_7 = q_as_7; assign final_exp_7 = q_fe_7;

endmodule


// =========================================================================
// 8b. PACK EXECUTE -- apply the shift decided by pack_decide_simd, extract
//     the final fraction bits, encode each format's word, and select the
//     active format. This is the "back half" of the original pack_simd,
//     on the far side of a pipeline register from the LZC/shift-amount
//     decision -- see pack_decide_simd for why the overflow and normal
//     cases share one shift (shift_src << act_shift) with the overflow
//     case simply forcing act_shift to 0.
// =========================================================================
module pack_execute_simd #(
    parameter E32 = 8,  parameter M32 = 23,  // FP32: 8-bit exponent, 23-bit fraction
    parameter E16 = 5,  parameter M16 = 10,  // FP16: 5-bit exponent, 10-bit fraction
    parameter E8  = 4,  parameter M8  = 3,  // FP8 (E4M3): 4-bit exponent, 3-bit fraction
    parameter E4  = 2,  parameter M4  = 1  // FP4 (E2M1): 2-bit exponent, 1-bit fraction
)(
    input [1:0] mode_control,

    // Per-lane decisions from pack_decide_simd (is_zero, shift_src, act_shift,
    // final_exp).
    input is_zero_0, input [M32+1:0] shift_src_0, input [4:0] act_shift_0, input [E32:0] final_exp_0,
    input is_zero_1, input [M16+1:0] shift_src_1, input [3:0] act_shift_1, input [E32:0] final_exp_1,
    input is_zero_2, input [M8+1:0]  shift_src_2, input [2:0] act_shift_2, input [E32:0] final_exp_2,
    input is_zero_3, input [M8+1:0]  shift_src_3, input [2:0] act_shift_3, input [E32:0] final_exp_3,
    input is_zero_4, input [M4+1:0]  shift_src_4, input [1:0] act_shift_4, input [E32:0] final_exp_4,
    input is_zero_5, input [M4+1:0]  shift_src_5, input [1:0] act_shift_5, input [E32:0] final_exp_5,
    input is_zero_6, input [M4+1:0]  shift_src_6, input [1:0] act_shift_6, input [E32:0] final_exp_6,
    input is_zero_7, input [M4+1:0]  shift_src_7, input [1:0] act_shift_7, input [E32:0] final_exp_7,

    // res_sign: sign of a non-zero result; zero_sign: sign to use if the result
    // is zero.
    input res_sign_0, res_sign_1, res_sign_2, res_sign_3,
    input res_sign_4, res_sign_5, res_sign_6, res_sign_7,
    input zero_sign_0, zero_sign_1, zero_sign_2, zero_sign_3,
    input zero_sign_4, zero_sign_5, zero_sign_6, zero_sign_7,

    output [31:0] result
);

    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings
    localparam WFP32 = M32+2; // 25
    localparam WFP16 = M16+2; // 12
    localparam WFP8  = M8+2;  // hidden + fraction + guard (5 bits for E4M3)
    localparam WFP4  = M4+2;  // 3

    // Note: FP4 saturates to +/-max-magnitude on overflow instead of
    // producing an "infinity" pattern like the other three formats do --
    // E2M1 has no dedicated Inf/NaN encoding in the standard (OCP MXFP4)
    // convention, so overflow is represented as {sign, 2'b11, 1'b1} (the
    // largest finite value, magnitude 6.0), matching typical MXFP4 clamp
    // behavior.

    // =====================================================================
    // LANE 0
    // =====================================================================
    // Normalize: left-shift shift_src by act_shift; a zero result skips the shift.
    wire [WFP32-1:0] sel_align0 = is_zero_0 ? {WFP32{1'b0}} : (shift_src_0 << act_shift_0);

    // inf_thresh: smallest exponent value that overflows the active format
    // (all-ones exponent for FP32/FP16; 4 for FP4, whose exponent field is
    // 2 bits). FP8 overflow is handled separately below.
    wire [7:0] inf_thresh0 = (mode_control==FP32) ? 8'd255 :
                              (mode_control==FP16) ? 8'd31 :
                              (mode_control==FP8)  ? 8'd15 : 8'd4;
    wire       res_ovf0    = (final_exp_0 >= {1'b0, inf_thresh0});

    // sel_align0 is normalized with the hidden bit at bit 24. The fraction is
    // the bits just below it; the guard bit and anything lower are truncated.
    // FP16/FP8/FP4 values are left-justified, so their fractions are the next
    // 10 / 3 / 1 bits.
    wire [22:0] fp32_frac0 = sel_align0[23:1];
    wire [9:0]  fp16_frac0 = sel_align0[23:14];
    wire [2:0]  fp8_frac0  = sel_align0[23:21];
    wire        fp4_frac0  = sel_align0[23];

    // OCP E4M3FN: exp field 1111 with frac 000-110 is an ordinary finite value
    // (up to 448); only frac==111 there is the reserved NaN codepoint. Genuine
    // magnitude overflow (final_exp_0 > 15, or landing exactly on the NaN
    // codepoint through rounding) saturates to the max finite value instead of
    // producing Inf, since E4M3FN has no Inf encoding.
    wire fp8_res_ovf0 = (final_exp_0 > 9'd15) | (final_exp_0 == 9'd15 & fp8_frac0 == 3'b111);

    // Flush-to-zero is folded into is_zero (and zero_sign) in the stage-3
    // register, so a flushed result simply takes the is_zero path here with no
    // extra logic after the shifter.
    // Result words, in priority order: zero, overflow (Inf for FP32/FP16,
    // saturation for FP8/FP4), normal.
    wire [31:0] fp32_word_0 = is_zero_0 ? {zero_sign_0, 31'b0} :
                               res_ovf0 ? {res_sign_0, 8'hFF, 23'h0} :
                                          {res_sign_0, final_exp_0[7:0], fp32_frac0};

    wire [15:0] fp16_word_0 = is_zero_0 ? {zero_sign_0, 15'b0} :
                               res_ovf0 ? {res_sign_0, 5'h1F, 10'h0} :
                                          {res_sign_0, final_exp_0[4:0], fp16_frac0};

    wire [7:0]  fp8_word_0  = is_zero_0    ? {zero_sign_0, 7'b0} :
                               fp8_res_ovf0 ? {res_sign_0, 4'hF, 3'h6} :  // saturate to +/-448
                                              {res_sign_0, final_exp_0[3:0], fp8_frac0};

    wire [3:0]  fp4_word_0  = is_zero_0 ? {zero_sign_0, 3'b0} :
                               res_ovf0 ? {res_sign_0, 2'b11, 1'b1} :
                                          {res_sign_0, final_exp_0[1:0], fp4_frac0};

    // =====================================================================
    // LANE 1
    // =====================================================================
    // Same structure as lane 0, with a 12-bit normalized value (hidden bit at
    // bit 11).
    wire [WFP16-1:0] sel_align1 = is_zero_1 ? {WFP16{1'b0}} : (shift_src_1 << act_shift_1);

    wire [4:0] inf_thresh1 = (mode_control==FP16) ? 5'd31 : (mode_control==FP8) ? 5'd15 : 5'd4;
    wire       res_ovf1    = (final_exp_1 >= {3'b0, inf_thresh1});

    wire [9:0] fp16_frac1 = sel_align1[10:1];
    wire [2:0] fp8_frac1  = sel_align1[10:8];
    wire       fp4_frac1  = sel_align1[10];

    wire fp8_res_ovf1 = (final_exp_1 > 9'd15) | (final_exp_1 == 9'd15 & fp8_frac1 == 3'b111);

    wire [15:0] fp16_word_1 = is_zero_1 ? {zero_sign_1, 15'b0} :
                               res_ovf1 ? {res_sign_1, 5'h1F, 10'h0} :
                                          {res_sign_1, final_exp_1[4:0], fp16_frac1};

    wire [7:0]  fp8_word_1  = is_zero_1    ? {zero_sign_1, 7'b0} :
                               fp8_res_ovf1 ? {res_sign_1, 4'hF, 3'h6} :  // saturate to +/-448
                                              {res_sign_1, final_exp_1[3:0], fp8_frac1};

    wire [3:0]  fp4_word_1  = is_zero_1 ? {zero_sign_1, 3'b0} :
                               res_ovf1 ? {res_sign_1, 2'b11, 1'b1} :
                                          {res_sign_1, final_exp_1[1:0], fp4_frac1};

    // =====================================================================
    // LANES 2 & 3
    // =====================================================================
    // EXEC_8_4(idx): result words for one FP8/FP4 lane (lanes 2 and 3), same
    // structure as lanes 0/1.
    `define EXEC_8_4(idx) \
        wire [WFP8-1:0] sel_align_``idx`` = is_zero_``idx`` ? {WFP8{1'b0}} : (shift_src_``idx`` << act_shift_``idx``); \
        wire [7:0] inf_thresh_``idx`` = (mode_control==FP8) ? 8'd15 : 8'd4; \
        wire res_ovf_``idx`` = (final_exp_``idx`` >= {1'b0, inf_thresh_``idx``}); \
        wire [M8-1:0] fp8_frac_``idx`` = sel_align_``idx``[WFP8-2 -: M8]; \
        wire       fp4_frac_``idx`` = sel_align_``idx``[WFP8-2]; \
        wire fp8_res_ovf_``idx`` = (final_exp_``idx`` > 9'd15) | (final_exp_``idx`` == 9'd15 & fp8_frac_``idx`` == 3'b111); \
        wire [7:0] fp8_word_``idx`` = is_zero_``idx``    ? {zero_sign_``idx``, 7'b0} : \
                                       fp8_res_ovf_``idx`` ? {res_sign_``idx``, 4'hF, 3'h6} : \
                                                              {res_sign_``idx``, final_exp_``idx``[3:0], fp8_frac_``idx``}; \
        wire [3:0] fp4_word_``idx`` = is_zero_``idx`` ? {zero_sign_``idx``, 3'b0} : \
                                       res_ovf_``idx`` ? {res_sign_``idx``, 2'b11, 1'b1} : \
                                                          {res_sign_``idx``, final_exp_``idx``[1:0], fp4_frac_``idx``};

    `EXEC_8_4(2)
    `EXEC_8_4(3)
    `undef EXEC_8_4

    // =====================================================================
    // LANES 4-7 (FP4-only)
    // =====================================================================
    // EXEC_4_ONLY(idx): FP4-only lanes (4..7); overflow saturates to
    // {sign, 2'b11, 1'b1}.
    `define EXEC_4_ONLY(idx) \
        wire [WFP4-1:0] sel_align4_``idx`` = is_zero_``idx`` ? {WFP4{1'b0}} : (shift_src_``idx`` << act_shift_``idx``); \
        wire [3:0] fp4_word_``idx`` = is_zero_``idx`` ? {zero_sign_``idx``, 3'b0} : \
                                       (final_exp_``idx`` >= 4) ? {res_sign_``idx``, 2'b11, 1'b1} : \
                                                                    {res_sign_``idx``, final_exp_``idx``[1:0], sel_align4_``idx``[WFP4-2:1]};

    `EXEC_4_ONLY(4)
    `EXEC_4_ONLY(5)
    `EXEC_4_ONLY(6)
    `EXEC_4_ONLY(7)
    `undef EXEC_4_ONLY

    // =====================================================================
    // FINAL MODE MUX
    // =====================================================================
    assign result = (mode_control == FP4)  ? {fp4_word_7, fp4_word_6, fp4_word_5, fp4_word_4,
                                               fp4_word_3, fp4_word_2, fp4_word_1, fp4_word_0} :
                     (mode_control == FP8)  ? {fp8_word_3, fp8_word_2, fp8_word_1, fp8_word_0} :
                     (mode_control == FP16) ? {fp16_word_1, fp16_word_0} :
                                               fp32_word_0;

endmodule


// =========================================================================
// 9. SPECIAL CASES DETECTION (Inf & NaN), 4 lanes x 3 modes
// =========================================================================
// match_k flags lane k as needing the Inf/NaN bypass; spec_res_k is the value
// to output in that case (the top level substitutes it for the arithmetic
// result of that lane).
//   FP32/FP16   : exponent all ones => Inf (fraction 0) or NaN (fraction != 0)
//   FP8 (E4M3FN): only S.1111.111 is NaN; there is no Inf, and the other
//                 exponent-1111 codes are ordinary finite values
//   FP4 (E2M1)  : no Inf/NaN encoding, so there is never a bypass
// Result for a matching lane: any NaN input, or infinities of opposite sign,
// gives a quiet NaN (FP32 0x7FC00000, FP16 0x7E00, FP8 0x7F); otherwise the
// result is the infinity operand.
module special_cases_simd (
    input  [1:0]  mode_control,
    input  [31:0] a, b,

    output match_0, output [31:0] spec_res_0,
    output match_1, output [15:0] spec_res_1,
    output match_2, output [7:0]  spec_res_2,
    output match_3, output [7:0]  spec_res_3,
    output match_4, output match_5, output match_6, output match_7 // always 0: FP4 has no Inf/NaN
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings

    // ---------------- LANE 0 (FP32 / FP16-lo / FP8-lo) ----------------
    wire        sign_a_0 = (mode_control==FP8) ? a[7]  : (mode_control==FP16) ? a[15] : a[31];
    wire        sign_b_0 = (mode_control==FP8) ? b[7]  : (mode_control==FP16) ? b[15] : b[31];
    wire [7:0]  exp_a_0  = (mode_control==FP8) ? {4'b0, a[6:3]}   : (mode_control==FP16) ? {3'b0, a[14:10]} : a[30:23];
    wire [7:0]  exp_b_0  = (mode_control==FP8) ? {4'b0, b[6:3]}   : (mode_control==FP16) ? {3'b0, b[14:10]} : b[30:23];
    wire [22:0] frac_a_0 = (mode_control==FP8) ? {20'b0, a[2:0]}  : (mode_control==FP16) ? {13'b0, a[9:0]}  : a[22:0];
    wire [22:0] frac_b_0 = (mode_control==FP8) ? {20'b0, b[2:0]}  : (mode_control==FP16) ? {13'b0, b[9:0]}  : b[22:0];

    // All-ones exponent and non-zero fraction tests for the active format.
    wire max_exp_a_0 = (mode_control==FP8) ? (exp_a_0[3:0]==4'b1111) :
                        (mode_control==FP16) ? (exp_a_0[4:0]==5'b11111) : (exp_a_0==8'hFF);
    wire max_exp_b_0 = (mode_control==FP8) ? (exp_b_0[3:0]==4'b1111) :
                        (mode_control==FP16) ? (exp_b_0[4:0]==5'b11111) : (exp_b_0==8'hFF);

    wire frac_nz_a_0 = (mode_control==FP8) ? (frac_a_0[2:0]!=0) : (mode_control==FP16) ? (frac_a_0[9:0]!=0) : (frac_a_0!=0);
    wire frac_nz_b_0 = (mode_control==FP8) ? (frac_b_0[2:0]!=0) : (mode_control==FP16) ? (frac_b_0[9:0]!=0) : (frac_b_0!=0);

    // OCP E4M3FN (FP8) has exactly one reserved codepoint, S.1111.111, which is
    // NaN; every other exp=1111 pattern (frac 000-110) is an ordinary finite
    // value and must fall through to normal add arithmetic, not be bypassed
    // here. There is no Inf encoding at all for FP8, so is_inf_* never fires
    // and overflow is handled as saturation inside pack_execute_simd instead.
    wire fp8_is_nan_a_0 = max_exp_a_0 & (frac_a_0[2:0]==3'b111);
    wire fp8_is_nan_b_0 = max_exp_b_0 & (frac_b_0[2:0]==3'b111);

    // NaN: all-ones exponent with non-zero fraction (FP32/FP16), or the single
    // reserved code (FP8). Inf: all-ones exponent with zero fraction (FP32/FP16).
    wire is_nan_a_0 = (mode_control==FP8) ? fp8_is_nan_a_0 : (max_exp_a_0 & frac_nz_a_0);
    wire is_nan_b_0 = (mode_control==FP8) ? fp8_is_nan_b_0 : (max_exp_b_0 & frac_nz_b_0);
    wire is_inf_a_0 = (mode_control==FP8) ? 1'b0 : (max_exp_a_0 & !frac_nz_a_0);
    wire is_inf_b_0 = (mode_control==FP8) ? 1'b0 : (max_exp_b_0 & !frac_nz_b_0);

    // FP4 (E2M1) has no dedicated Inf/NaN encoding in the standard convention
    // used here, so the special-case bypass never triggers for FP4 -- pack_simd
    // handles FP4 overflow by saturating to +/-max-magnitude instead.
    assign match_0 = (mode_control == FP4) ? 1'b0 :
                      (mode_control == FP8) ? (fp8_is_nan_a_0 | fp8_is_nan_b_0) :
                                               (max_exp_a_0 | max_exp_b_0);

    // Quiet-NaN and signed-infinity encodings for the active format.
    wire [31:0] nan_val_0   = (mode_control==FP8) ? {24'b0, 8'h7F} : (mode_control==FP16) ? {16'b0, 16'h7E00} : 32'h7FC0_0000;
    wire [31:0] inf_a_val_0 = (mode_control==FP8) ? 32'h0 :  // unused: match_0 only fires on NaN for FP8
                               (mode_control==FP16) ? {16'b0, sign_a_0, 15'h7C00} :
                                                       {sign_a_0, 8'hFF, 23'h0};
    wire [31:0] inf_b_val_0 = (mode_control==FP8) ? 32'h0 :
                               (mode_control==FP16) ? {16'b0, sign_b_0, 15'h7C00} :
                                                       {sign_b_0, 8'hFF, 23'h0};

    // NaN if either input is NaN or the infinities have opposite signs;
    // otherwise a's infinity if a is Inf, else b's.
    assign spec_res_0 = (is_nan_a_0 | is_nan_b_0) ? nan_val_0 :
                         (is_inf_a_0 & is_inf_b_0 & (sign_a_0 != sign_b_0)) ? nan_val_0 :
                         (is_inf_a_0) ? inf_a_val_0 : inf_b_val_0;

    // ---------------- LANE 1 (FP16-hi / FP8 lane1) ----------------
    wire       sign_a_1 = (mode_control==FP8) ? a[15] : a[31];
    wire       sign_b_1 = (mode_control==FP8) ? b[15] : b[31];
    wire [4:0] exp_a_1  = (mode_control==FP8) ? {1'b0, a[14:11]} : a[30:26];
    wire [4:0] exp_b_1  = (mode_control==FP8) ? {1'b0, b[14:11]} : b[30:26];
    wire [9:0] frac_a_1 = (mode_control==FP8) ? {7'b0, a[10:8]}  : a[25:16];
    wire [9:0] frac_b_1 = (mode_control==FP8) ? {7'b0, b[10:8]}  : b[25:16];

    wire max_exp_a_1 = (mode_control==FP8) ? (exp_a_1[3:0]==4'b1111) : (exp_a_1==5'b11111);
    wire max_exp_b_1 = (mode_control==FP8) ? (exp_b_1[3:0]==4'b1111) : (exp_b_1==5'b11111);
    wire frac_nz_a_1 = (mode_control==FP8) ? (frac_a_1[2:0]!=0) : (frac_a_1!=0);
    wire frac_nz_b_1 = (mode_control==FP8) ? (frac_b_1[2:0]!=0) : (frac_b_1!=0);

    wire fp8_is_nan_a_1 = max_exp_a_1 & (frac_a_1[2:0]==3'b111);
    wire fp8_is_nan_b_1 = max_exp_b_1 & (frac_b_1[2:0]==3'b111);

    wire is_nan_a_1 = (mode_control==FP8) ? fp8_is_nan_a_1 : (max_exp_a_1 & frac_nz_a_1);
    wire is_nan_b_1 = (mode_control==FP8) ? fp8_is_nan_b_1 : (max_exp_b_1 & frac_nz_b_1);
    wire is_inf_a_1 = (mode_control==FP8) ? 1'b0 : (max_exp_a_1 & !frac_nz_a_1);
    wire is_inf_b_1 = (mode_control==FP8) ? 1'b0 : (max_exp_b_1 & !frac_nz_b_1);

    assign match_1 = (mode_control == FP16) ? (max_exp_a_1 | max_exp_b_1) :
                      (mode_control == FP8)  ? (fp8_is_nan_a_1 | fp8_is_nan_b_1) : 1'b0;

    wire [15:0] nan_val_1   = (mode_control==FP8) ? {8'b0, 8'h7F} : 16'h7E00;
    wire [15:0] inf_a_val_1 = (mode_control==FP8) ? 16'h0 : {sign_a_1, 15'h7C00};
    wire [15:0] inf_b_val_1 = (mode_control==FP8) ? 16'h0 : {sign_b_1, 15'h7C00};

    // Same selection rule as lane 0.
    assign spec_res_1 = (is_nan_a_1 | is_nan_b_1) ? nan_val_1 :
                         (is_inf_a_1 & is_inf_b_1 & (sign_a_1 != sign_b_1)) ? nan_val_1 :
                         (is_inf_a_1) ? inf_a_val_1 : inf_b_val_1;

    // ---------------- LANE 2 (FP8 lane2 only) ----------------
    wire       sign_a_2 = a[23];
    wire       sign_b_2 = b[23];
    wire [3:0] exp_a_2  = a[22:19];
    wire [3:0] exp_b_2  = b[22:19];
    wire [2:0] frac_a_2 = a[18:16];
    wire [2:0] frac_b_2 = b[18:16];

    wire max_exp_a_2 = (exp_a_2==4'b1111);
    wire max_exp_b_2 = (exp_b_2==4'b1111);
    // OCP E4M3FN: only frac==111 at max exponent is NaN; frac 000-110 there are
    // ordinary finite values (up to 448) and must go through normal arithmetic.
    // No Inf encoding exists, so overflow saturates in pack_execute_simd instead.
    wire is_nan_a_2 = max_exp_a_2 & (frac_a_2==3'b111);
    wire is_nan_b_2 = max_exp_b_2 & (frac_b_2==3'b111);

    assign match_2 = (mode_control == FP8) & (is_nan_a_2 | is_nan_b_2);
    wire [7:0] nan_val_2 = 8'h7F;

    assign spec_res_2 = nan_val_2;  // match_2 only ever fires on NaN input now

    // ---------------- LANE 3 (FP8 lane3 only) ----------------
    wire       sign_a_3 = a[31];
    wire       sign_b_3 = b[31];
    wire [3:0] exp_a_3  = a[30:27];
    wire [3:0] exp_b_3  = b[30:27];
    wire [2:0] frac_a_3 = a[26:24];
    wire [2:0] frac_b_3 = b[26:24];

    wire max_exp_a_3 = (exp_a_3==4'b1111);
    wire max_exp_b_3 = (exp_b_3==4'b1111);
    wire is_nan_a_3 = max_exp_a_3 & (frac_a_3==3'b111);
    wire is_nan_b_3 = max_exp_b_3 & (frac_b_3==3'b111);

    assign match_3 = (mode_control == FP8) & (is_nan_a_3 | is_nan_b_3);
    wire [7:0] nan_val_3 = 8'h7F;

    assign spec_res_3 = nan_val_3;  // match_3 only ever fires on NaN input now

    // FP4 lanes 4..7 never match (no Inf/NaN encoding).
    assign match_4 = 1'b0;
    assign match_5 = 1'b0;
    assign match_6 = 1'b0;
    assign match_7 = 1'b0;

endmodule


// =========================================================================
// 10. TOP-LEVEL ADDER MODULE -- 5-REGISTER PIPELINE
// =========================================================================
// The datapath is cut so that each of the deepest blocks (alignment barrel
// shifter, mantissa adder, leading-zero count, normalization barrel shifter)
// gets its own cycle, which shortens the critical path and raises the
// achievable clock frequency at the cost of latency.
//
//   STAGE 1  (combinational) unpack_simd, exp_compare_simd, exp_diff_simd,
//            signed_xor_simd, shifter_prep_simd; special_cases_simd also
//            runs here since it needs only the raw operands
//   REGISTER 1
//   STAGE 2a alignment shift of the smaller operand (shared_shifter32)
//   REGISTER 1b
//   STAGE 2b mantissa add / subtract (add_slice_simd)
//   REGISTER 2
//   STAGE 3a pack_decide_simd, first half: overflow / zero detect,
//            leading-zero count, exponent - 1 (lanes 2..7 finish here)
//   REGISTER inside pack_decide_simd (REGISTER 2b delays the sideband
//            signals by one cycle to stay aligned with it)
//   STAGE 3b pack_decide_simd, second half: shift clamp, final exponent and
//            flush-to-zero decision for lanes 0 and 1
//   REGISTER 3
//   STAGE 4  pack_execute_simd (normalization shift, result encoding), then
//            the Inf/NaN bypass and the final format mux
//
// Latency: five register stages (R1, R1b, R2, the pack_decide_simd internal
// registers, R3). The result is combinational from R3, so it is valid after
// the fifth enabled clock edge following the operands.
// All registers are gated by the clock enable ce.
// =========================================================================
module fp_adder_simd #(
    parameter FTZ_EN = 1,   // flush-to-zero for subnormal FP32 / FP16 results
    parameter E32 = 8,  parameter M32 = 23,  // FP32: 8-bit exponent, 23-bit fraction
    parameter E16 = 5,  parameter M16 = 10,  // FP16: 5-bit exponent, 10-bit fraction
    parameter E8  = 4,  parameter M8  = 3,  // FP8 (E4M3): 4-bit exponent, 3-bit fraction
    parameter E4  = 2,  parameter M4  = 1  // FP4 (E2M1): 2-bit exponent, 1-bit fraction
)(
    input  wire       clk,  // clock
    input  wire       ce,  // clock enable for every pipeline register
    input  wire [1:0] mode_control,   // 00: FP32, 01: 2xFP16, 10: 4xFP8, 11: 8xFP4
    input  wire [31:0] a,  // packed operand A (format set by mode_control)
    input  wire [31:0] b,  // packed operand B
    output wire [31:0] result  // packed result, same lane layout as a/b
);

    // ================= STAGE 1 wires =================
    // Unpacked operand fields per lane: sign, biased exponent, fraction.
    wire signed_a_0, signed_b_0, signed_a_1, signed_b_1;
    wire signed_a_2, signed_b_2, signed_a_3, signed_b_3;
    wire signed_a_4, signed_b_4, signed_a_5, signed_b_5;
    wire signed_a_6, signed_b_6, signed_a_7, signed_b_7;
    wire [E32-1:0] exp_a_0, exp_b_0, exp_a_1, exp_b_1, exp_a_2, exp_b_2, exp_a_3, exp_b_3;
    wire [E32-1:0] exp_a_4, exp_b_4, exp_a_5, exp_b_5, exp_a_6, exp_b_6, exp_a_7, exp_b_7;
    wire [M32-1:0] frac_a_0, frac_b_0, frac_a_1, frac_b_1, frac_a_2, frac_b_2, frac_a_3, frac_b_3;
    wire [M32-1:0] frac_a_4, frac_b_4, frac_a_5, frac_b_5, frac_a_6, frac_b_6, frac_a_7, frac_b_7;
    // Exponent compare outputs per lane: effective exponents (0 replaced by 1),
    // hidden-bit flags (exponent non-zero) and "a has the larger exponent" flags.
    wire [E32-1:0] eff_exp_a_0, eff_exp_b_0, eff_exp_a_1, eff_exp_b_1;
    wire [E32-1:0] eff_exp_a_2, eff_exp_b_2, eff_exp_a_3, eff_exp_b_3;
    wire [E32-1:0] eff_exp_a_4, eff_exp_b_4, eff_exp_a_5, eff_exp_b_5;
    wire [E32-1:0] eff_exp_a_6, eff_exp_b_6, eff_exp_a_7, eff_exp_b_7;
    wire exp_a_notzero_0, exp_b_notzero_0, exp_a_notzero_1, exp_b_notzero_1;
    wire exp_a_notzero_2, exp_b_notzero_2, exp_a_notzero_3, exp_b_notzero_3;
    wire exp_a_notzero_4, exp_b_notzero_4, exp_a_notzero_5, exp_b_notzero_5;
    wire exp_a_notzero_6, exp_b_notzero_6, exp_a_notzero_7, exp_b_notzero_7;
    wire exp_a_larger_0, exp_a_larger_1, exp_a_larger_2, exp_a_larger_3;
    wire exp_a_larger_4, exp_a_larger_5, exp_a_larger_6, exp_a_larger_7;
    // Common (larger) exponent and alignment shift amount per lane.
    wire [E32-1:0] s1_exp_common_0, s1_exp_common_1, s1_exp_common_2, s1_exp_common_3;
    wire [E32-1:0] s1_exp_common_4, s1_exp_common_5, s1_exp_common_6, s1_exp_common_7;
    wire [E32-1:0] s1_exp_diff_0, s1_exp_diff_1, s1_exp_diff_2, s1_exp_diff_3;
    wire [E32-1:0] s1_exp_diff_4, s1_exp_diff_5, s1_exp_diff_6, s1_exp_diff_7;
    // Effective-subtract flag, result sign and zero-result sign per lane.
    wire s1_signed_xor_0, s1_signed_xor_1, s1_signed_xor_2, s1_signed_xor_3;
    wire s1_signed_xor_4, s1_signed_xor_5, s1_signed_xor_6, s1_signed_xor_7;
    wire s1_res_sign_0, s1_res_sign_1, s1_res_sign_2, s1_res_sign_3;
    wire s1_res_sign_4, s1_res_sign_5, s1_res_sign_6, s1_res_sign_7;
    wire s1_zero_sign_0, s1_zero_sign_1, s1_zero_sign_2, s1_zero_sign_3;
    wire s1_zero_sign_4, s1_zero_sign_5, s1_zero_sign_6, s1_zero_sign_7;
    // Nibble-aligned larger-operand bus and operand-to-align bus.
    wire [31:0] s1_base32, s1_to_shift32;
    // Inf/NaN bypass flags and the replacement values.
    wire s1_match_0, s1_match_1, s1_match_2, s1_match_3, s1_match_4, s1_match_5, s1_match_6, s1_match_7;
    wire [31:0] s1_spec_res_0;
    wire [15:0] s1_spec_res_1;
    wire [7:0]  s1_spec_res_2, s1_spec_res_3;

    // Split a and b into per-lane sign / exponent / fraction.
    unpack_simd #( .E32(E32), .M32(M32), .E16(E16), .M16(M16), .E8(E8), .M8(M8), .E4(E4), .M4(M4) ) u_unpack (
        .mode_control(mode_control), .a(a), .b(b),
        .signed_a_0(signed_a_0), .signed_b_0(signed_b_0),
        .signed_a_1(signed_a_1), .signed_b_1(signed_b_1),
        .signed_a_2(signed_a_2), .signed_b_2(signed_b_2),
        .signed_a_3(signed_a_3), .signed_b_3(signed_b_3),
        .signed_a_4(signed_a_4), .signed_b_4(signed_b_4),
        .signed_a_5(signed_a_5), .signed_b_5(signed_b_5),
        .signed_a_6(signed_a_6), .signed_b_6(signed_b_6),
        .signed_a_7(signed_a_7), .signed_b_7(signed_b_7),
        .exp_a_0(exp_a_0), .exp_b_0(exp_b_0), .exp_a_1(exp_a_1), .exp_b_1(exp_b_1),
        .exp_a_2(exp_a_2), .exp_b_2(exp_b_2), .exp_a_3(exp_a_3), .exp_b_3(exp_b_3),
        .exp_a_4(exp_a_4), .exp_b_4(exp_b_4), .exp_a_5(exp_a_5), .exp_b_5(exp_b_5),
        .exp_a_6(exp_a_6), .exp_b_6(exp_b_6), .exp_a_7(exp_a_7), .exp_b_7(exp_b_7),
        .frac_a_0(frac_a_0), .frac_b_0(frac_b_0), .frac_a_1(frac_a_1), .frac_b_1(frac_b_1),
        .frac_a_2(frac_a_2), .frac_b_2(frac_b_2), .frac_a_3(frac_a_3), .frac_b_3(frac_b_3),
        .frac_a_4(frac_a_4), .frac_b_4(frac_b_4), .frac_a_5(frac_a_5), .frac_b_5(frac_b_5),
        .frac_a_6(frac_a_6), .frac_b_6(frac_b_6), .frac_a_7(frac_a_7), .frac_b_7(frac_b_7)
    );

    // Per-lane exponent compare: hidden-bit flags, effective exponents and
    // which operand has the larger exponent.
    exp_compare_simd #( .E(E32) ) u_exp_compare (
        .exp_a_0(exp_a_0), .exp_b_0(exp_b_0), .exp_a_1(exp_a_1), .exp_b_1(exp_b_1),
        .exp_a_2(exp_a_2), .exp_b_2(exp_b_2), .exp_a_3(exp_a_3), .exp_b_3(exp_b_3),
        .exp_a_4(exp_a_4), .exp_b_4(exp_b_4), .exp_a_5(exp_a_5), .exp_b_5(exp_b_5),
        .exp_a_6(exp_a_6), .exp_b_6(exp_b_6), .exp_a_7(exp_a_7), .exp_b_7(exp_b_7),
        .exp_a_notzero_0(exp_a_notzero_0), .exp_b_notzero_0(exp_b_notzero_0), .exp_a_larger_0(exp_a_larger_0),
        .exp_a_notzero_1(exp_a_notzero_1), .exp_b_notzero_1(exp_b_notzero_1), .exp_a_larger_1(exp_a_larger_1),
        .exp_a_notzero_2(exp_a_notzero_2), .exp_b_notzero_2(exp_b_notzero_2), .exp_a_larger_2(exp_a_larger_2),
        .exp_a_notzero_3(exp_a_notzero_3), .exp_b_notzero_3(exp_b_notzero_3), .exp_a_larger_3(exp_a_larger_3),
        .exp_a_notzero_4(exp_a_notzero_4), .exp_b_notzero_4(exp_b_notzero_4), .exp_a_larger_4(exp_a_larger_4),
        .exp_a_notzero_5(exp_a_notzero_5), .exp_b_notzero_5(exp_b_notzero_5), .exp_a_larger_5(exp_a_larger_5),
        .exp_a_notzero_6(exp_a_notzero_6), .exp_b_notzero_6(exp_b_notzero_6), .exp_a_larger_6(exp_a_larger_6),
        .exp_a_notzero_7(exp_a_notzero_7), .exp_b_notzero_7(exp_b_notzero_7), .exp_a_larger_7(exp_a_larger_7),
        .eff_exp_a_0(eff_exp_a_0), .eff_exp_b_0(eff_exp_b_0), .eff_exp_a_1(eff_exp_a_1), .eff_exp_b_1(eff_exp_b_1),
        .eff_exp_a_2(eff_exp_a_2), .eff_exp_b_2(eff_exp_b_2), .eff_exp_a_3(eff_exp_a_3), .eff_exp_b_3(eff_exp_b_3),
        .eff_exp_a_4(eff_exp_a_4), .eff_exp_b_4(eff_exp_b_4), .eff_exp_a_5(eff_exp_a_5), .eff_exp_b_5(eff_exp_b_5),
        .eff_exp_a_6(eff_exp_a_6), .eff_exp_b_6(eff_exp_b_6), .eff_exp_a_7(eff_exp_a_7), .eff_exp_b_7(eff_exp_b_7)
    );

    // Common exponent and |exponent difference| per lane (works on the
    // effective exponents).
    exp_diff_simd #( .E(E32) ) u_exp_diff (
        .exp_a_0(eff_exp_a_0), .exp_b_0(eff_exp_b_0), .exp_a_1(eff_exp_a_1), .exp_b_1(eff_exp_b_1),
        .exp_a_2(eff_exp_a_2), .exp_b_2(eff_exp_b_2), .exp_a_3(eff_exp_a_3), .exp_b_3(eff_exp_b_3),
        .exp_a_4(eff_exp_a_4), .exp_b_4(eff_exp_b_4), .exp_a_5(eff_exp_a_5), .exp_b_5(eff_exp_b_5),
        .exp_a_6(eff_exp_a_6), .exp_b_6(eff_exp_b_6), .exp_a_7(eff_exp_a_7), .exp_b_7(eff_exp_b_7),
        .exp_a_larger_0(exp_a_larger_0), .exp_a_larger_1(exp_a_larger_1),
        .exp_a_larger_2(exp_a_larger_2), .exp_a_larger_3(exp_a_larger_3),
        .exp_a_larger_4(exp_a_larger_4), .exp_a_larger_5(exp_a_larger_5),
        .exp_a_larger_6(exp_a_larger_6), .exp_a_larger_7(exp_a_larger_7),
        .exp_common_0(s1_exp_common_0), .exp_diff_0(s1_exp_diff_0),
        .exp_common_1(s1_exp_common_1), .exp_diff_1(s1_exp_diff_1),
        .exp_common_2(s1_exp_common_2), .exp_diff_2(s1_exp_diff_2),
        .exp_common_3(s1_exp_common_3), .exp_diff_3(s1_exp_diff_3),
        .exp_common_4(s1_exp_common_4), .exp_diff_4(s1_exp_diff_4),
        .exp_common_5(s1_exp_common_5), .exp_diff_5(s1_exp_diff_5),
        .exp_common_6(s1_exp_common_6), .exp_diff_6(s1_exp_diff_6),
        .exp_common_7(s1_exp_common_7), .exp_diff_7(s1_exp_diff_7)
    );

    // Effective-subtract flag per active lane.
    signed_xor_simd u_signed_xor (
        .mode_control(mode_control),
        .signed_a_0(signed_a_0), .signed_b_0(signed_b_0), .signed_a_1(signed_a_1), .signed_b_1(signed_b_1),
        .signed_a_2(signed_a_2), .signed_b_2(signed_b_2), .signed_a_3(signed_a_3), .signed_b_3(signed_b_3),
        .signed_a_4(signed_a_4), .signed_b_4(signed_b_4), .signed_a_5(signed_a_5), .signed_b_5(signed_b_5),
        .signed_a_6(signed_a_6), .signed_b_6(signed_b_6), .signed_a_7(signed_a_7), .signed_b_7(signed_b_7),
        .signed_xor_0(s1_signed_xor_0), .signed_xor_1(s1_signed_xor_1),
        .signed_xor_2(s1_signed_xor_2), .signed_xor_3(s1_signed_xor_3),
        .signed_xor_4(s1_signed_xor_4), .signed_xor_5(s1_signed_xor_5),
        .signed_xor_6(s1_signed_xor_6), .signed_xor_7(s1_signed_xor_7)
    );

    // Order the operands by magnitude, format the mantissas and build the
    // composite base / to-shift buses.
    shifter_prep_simd #( .E(E32), .M(M32), .M16(M16), .M8(M8), .M4(M4) ) u_shifter_prep (
        .mode_control(mode_control),
        .exp_diff_0(s1_exp_diff_0), .exp_diff_1(s1_exp_diff_1), .exp_diff_2(s1_exp_diff_2), .exp_diff_3(s1_exp_diff_3),
        .exp_diff_4(s1_exp_diff_4), .exp_diff_5(s1_exp_diff_5), .exp_diff_6(s1_exp_diff_6), .exp_diff_7(s1_exp_diff_7),
        .signed_a_0(signed_a_0), .signed_b_0(signed_b_0), .signed_a_1(signed_a_1), .signed_b_1(signed_b_1),
        .signed_a_2(signed_a_2), .signed_b_2(signed_b_2), .signed_a_3(signed_a_3), .signed_b_3(signed_b_3),
        .signed_a_4(signed_a_4), .signed_b_4(signed_b_4), .signed_a_5(signed_a_5), .signed_b_5(signed_b_5),
        .signed_a_6(signed_a_6), .signed_b_6(signed_b_6), .signed_a_7(signed_a_7), .signed_b_7(signed_b_7),
        .exp_a_notzero_0(exp_a_notzero_0), .exp_b_notzero_0(exp_b_notzero_0),
        .exp_a_notzero_1(exp_a_notzero_1), .exp_b_notzero_1(exp_b_notzero_1),
        .exp_a_notzero_2(exp_a_notzero_2), .exp_b_notzero_2(exp_b_notzero_2),
        .exp_a_notzero_3(exp_a_notzero_3), .exp_b_notzero_3(exp_b_notzero_3),
        .exp_a_notzero_4(exp_a_notzero_4), .exp_b_notzero_4(exp_b_notzero_4),
        .exp_a_notzero_5(exp_a_notzero_5), .exp_b_notzero_5(exp_b_notzero_5),
        .exp_a_notzero_6(exp_a_notzero_6), .exp_b_notzero_6(exp_b_notzero_6),
        .exp_a_notzero_7(exp_a_notzero_7), .exp_b_notzero_7(exp_b_notzero_7),
        .exp_a_larger_0(exp_a_larger_0), .exp_a_larger_1(exp_a_larger_1),
        .exp_a_larger_2(exp_a_larger_2), .exp_a_larger_3(exp_a_larger_3),
        .exp_a_larger_4(exp_a_larger_4), .exp_a_larger_5(exp_a_larger_5),
        .exp_a_larger_6(exp_a_larger_6), .exp_a_larger_7(exp_a_larger_7),
        .frac_a_0(frac_a_0), .frac_b_0(frac_b_0), .frac_a_1(frac_a_1), .frac_b_1(frac_b_1),
        .frac_a_2(frac_a_2), .frac_b_2(frac_b_2), .frac_a_3(frac_a_3), .frac_b_3(frac_b_3),
        .frac_a_4(frac_a_4), .frac_b_4(frac_b_4), .frac_a_5(frac_a_5), .frac_b_5(frac_b_5),
        .frac_a_6(frac_a_6), .frac_b_6(frac_b_6), .frac_a_7(frac_a_7), .frac_b_7(frac_b_7),
        .res_sign_0(s1_res_sign_0), .res_sign_1(s1_res_sign_1), .res_sign_2(s1_res_sign_2), .res_sign_3(s1_res_sign_3),
        .res_sign_4(s1_res_sign_4), .res_sign_5(s1_res_sign_5), .res_sign_6(s1_res_sign_6), .res_sign_7(s1_res_sign_7),
        .zero_sign_0(s1_zero_sign_0), .zero_sign_1(s1_zero_sign_1), .zero_sign_2(s1_zero_sign_2), .zero_sign_3(s1_zero_sign_3),
        .zero_sign_4(s1_zero_sign_4), .zero_sign_5(s1_zero_sign_5), .zero_sign_6(s1_zero_sign_6), .zero_sign_7(s1_zero_sign_7),
        .base32(s1_base32), .to_shift32(s1_to_shift32)
    );

    // Detect Inf / NaN operands from the raw inputs.
    special_cases_simd u_spec (
        .mode_control(mode_control), .a(a), .b(b),
        .match_0(s1_match_0), .spec_res_0(s1_spec_res_0),
        .match_1(s1_match_1), .spec_res_1(s1_spec_res_1),
        .match_2(s1_match_2), .spec_res_2(s1_spec_res_2),
        .match_3(s1_match_3), .spec_res_3(s1_spec_res_3),
        .match_4(s1_match_4), .match_5(s1_match_5), .match_6(s1_match_6), .match_7(s1_match_7)
    );

    // ================= REGISTER 1 =================
    // shreg_extract = "no" keeps the pipeline registers as plain flip-flops
    // instead of letting synthesis merge them into shift-register (SRL) cells.
    (* shreg_extract = "no" *) reg [31:0] r1_base32, r1_to_shift32;
    (* shreg_extract = "no" *) reg [E32-1:0] r1_exp_diff_0, r1_exp_diff_1, r1_exp_diff_2, r1_exp_diff_3;
    (* shreg_extract = "no" *) reg [E32-1:0] r1_exp_diff_4, r1_exp_diff_5, r1_exp_diff_6, r1_exp_diff_7;
    (* shreg_extract = "no" *) reg r1_signed_xor_0, r1_signed_xor_1, r1_signed_xor_2, r1_signed_xor_3;
    (* shreg_extract = "no" *) reg r1_signed_xor_4, r1_signed_xor_5, r1_signed_xor_6, r1_signed_xor_7;
    (* shreg_extract = "no" *) reg r1_res_sign_0, r1_res_sign_1, r1_res_sign_2, r1_res_sign_3;
    (* shreg_extract = "no" *) reg r1_res_sign_4, r1_res_sign_5, r1_res_sign_6, r1_res_sign_7;
    (* shreg_extract = "no" *) reg r1_zero_sign_0, r1_zero_sign_1, r1_zero_sign_2, r1_zero_sign_3;
    (* shreg_extract = "no" *) reg r1_zero_sign_4, r1_zero_sign_5, r1_zero_sign_6, r1_zero_sign_7;
    (* shreg_extract = "no" *) reg [E32-1:0] r1_exp_common_0, r1_exp_common_1, r1_exp_common_2, r1_exp_common_3;
    (* shreg_extract = "no" *) reg [E32-1:0] r1_exp_common_4, r1_exp_common_5, r1_exp_common_6, r1_exp_common_7;
    (* shreg_extract = "no" *) reg [1:0] r1_mode;
    (* shreg_extract = "no" *) reg r1_match_0, r1_match_1, r1_match_2, r1_match_3, r1_match_4, r1_match_5, r1_match_6, r1_match_7;
    (* shreg_extract = "no" *) reg [31:0] r1_spec_res_0;
    (* shreg_extract = "no" *) reg [15:0] r1_spec_res_1;
    (* shreg_extract = "no" *) reg [7:0]  r1_spec_res_2, r1_spec_res_3;

    always @(posedge clk) begin
        if (ce) begin
            r1_base32     <= s1_base32;
            r1_to_shift32 <= s1_to_shift32;
            r1_exp_diff_0 <= s1_exp_diff_0; r1_exp_diff_1 <= s1_exp_diff_1;
            r1_exp_diff_2 <= s1_exp_diff_2; r1_exp_diff_3 <= s1_exp_diff_3;
            r1_exp_diff_4 <= s1_exp_diff_4; r1_exp_diff_5 <= s1_exp_diff_5;
            r1_exp_diff_6 <= s1_exp_diff_6; r1_exp_diff_7 <= s1_exp_diff_7;
            r1_signed_xor_0 <= s1_signed_xor_0; r1_signed_xor_1 <= s1_signed_xor_1;
            r1_signed_xor_2 <= s1_signed_xor_2; r1_signed_xor_3 <= s1_signed_xor_3;
            r1_signed_xor_4 <= s1_signed_xor_4; r1_signed_xor_5 <= s1_signed_xor_5;
            r1_signed_xor_6 <= s1_signed_xor_6; r1_signed_xor_7 <= s1_signed_xor_7;
            r1_res_sign_0 <= s1_res_sign_0; r1_res_sign_1 <= s1_res_sign_1;
            r1_res_sign_2 <= s1_res_sign_2; r1_res_sign_3 <= s1_res_sign_3;
            r1_res_sign_4 <= s1_res_sign_4; r1_res_sign_5 <= s1_res_sign_5;
            r1_res_sign_6 <= s1_res_sign_6; r1_res_sign_7 <= s1_res_sign_7;
            r1_zero_sign_0 <= s1_zero_sign_0; r1_zero_sign_1 <= s1_zero_sign_1;
            r1_zero_sign_2 <= s1_zero_sign_2; r1_zero_sign_3 <= s1_zero_sign_3;
            r1_zero_sign_4 <= s1_zero_sign_4; r1_zero_sign_5 <= s1_zero_sign_5;
            r1_zero_sign_6 <= s1_zero_sign_6; r1_zero_sign_7 <= s1_zero_sign_7;
            r1_exp_common_0 <= s1_exp_common_0; r1_exp_common_1 <= s1_exp_common_1;
            r1_exp_common_2 <= s1_exp_common_2; r1_exp_common_3 <= s1_exp_common_3;
            r1_exp_common_4 <= s1_exp_common_4; r1_exp_common_5 <= s1_exp_common_5;
            r1_exp_common_6 <= s1_exp_common_6; r1_exp_common_7 <= s1_exp_common_7;
            r1_mode <= mode_control;
            r1_match_0 <= s1_match_0; r1_match_1 <= s1_match_1; r1_match_2 <= s1_match_2; r1_match_3 <= s1_match_3;
            r1_match_4 <= s1_match_4; r1_match_5 <= s1_match_5; r1_match_6 <= s1_match_6; r1_match_7 <= s1_match_7;
            r1_spec_res_0 <= s1_spec_res_0; r1_spec_res_1 <= s1_spec_res_1;
            r1_spec_res_2 <= s1_spec_res_2; r1_spec_res_3 <= s1_spec_res_3;
        end
    end

    // ================= STAGE 2a: alignment shift =================
    wire [31:0] s2a_shifted32;
    // Right-shift the smaller operand by the exponent difference (lane-segmented).
    shared_shifter32 #( .E(E32), .M8(M8) ) u_shared_shifter (
        .mode_control(r1_mode), .data_in(r1_to_shift32),
        .exp_diff_0(r1_exp_diff_0), .exp_diff_1(r1_exp_diff_1), .exp_diff_2(r1_exp_diff_2), .exp_diff_3(r1_exp_diff_3),
        .exp_diff_4(r1_exp_diff_4), .exp_diff_5(r1_exp_diff_5), .exp_diff_6(r1_exp_diff_6), .exp_diff_7(r1_exp_diff_7),
        .data_out(s2a_shifted32)
    );

    // ================= REGISTER 1b (between alignment shift and add) =================
    (* shreg_extract = "no" *) reg [31:0] r1b_base32, r1b_shifted32;
    (* shreg_extract = "no" *) reg r1b_signed_xor_0, r1b_signed_xor_1, r1b_signed_xor_2, r1b_signed_xor_3;
    (* shreg_extract = "no" *) reg r1b_signed_xor_4, r1b_signed_xor_5, r1b_signed_xor_6, r1b_signed_xor_7;
    (* shreg_extract = "no" *) reg r1b_res_sign_0, r1b_res_sign_1, r1b_res_sign_2, r1b_res_sign_3;
    (* shreg_extract = "no" *) reg r1b_res_sign_4, r1b_res_sign_5, r1b_res_sign_6, r1b_res_sign_7;
    (* shreg_extract = "no" *) reg r1b_zero_sign_0, r1b_zero_sign_1, r1b_zero_sign_2, r1b_zero_sign_3;
    (* shreg_extract = "no" *) reg r1b_zero_sign_4, r1b_zero_sign_5, r1b_zero_sign_6, r1b_zero_sign_7;
    (* shreg_extract = "no" *) reg [E32-1:0] r1b_exp_common_0, r1b_exp_common_1, r1b_exp_common_2, r1b_exp_common_3;
    (* shreg_extract = "no" *) reg [E32-1:0] r1b_exp_common_4, r1b_exp_common_5, r1b_exp_common_6, r1b_exp_common_7;
    (* shreg_extract = "no" *) reg [1:0] r1b_mode;
    (* shreg_extract = "no" *) reg r1b_match_0, r1b_match_1, r1b_match_2, r1b_match_3, r1b_match_4, r1b_match_5, r1b_match_6, r1b_match_7;
    (* shreg_extract = "no" *) reg [31:0] r1b_spec_res_0;
    (* shreg_extract = "no" *) reg [15:0] r1b_spec_res_1;
    (* shreg_extract = "no" *) reg [7:0]  r1b_spec_res_2, r1b_spec_res_3;

    always @(posedge clk) begin
        if (ce) begin
            r1b_base32 <= r1_base32; r1b_shifted32 <= s2a_shifted32;
            r1b_signed_xor_0 <= r1_signed_xor_0; r1b_signed_xor_1 <= r1_signed_xor_1;
            r1b_signed_xor_2 <= r1_signed_xor_2; r1b_signed_xor_3 <= r1_signed_xor_3;
            r1b_signed_xor_4 <= r1_signed_xor_4; r1b_signed_xor_5 <= r1_signed_xor_5;
            r1b_signed_xor_6 <= r1_signed_xor_6; r1b_signed_xor_7 <= r1_signed_xor_7;
            r1b_res_sign_0 <= r1_res_sign_0; r1b_res_sign_1 <= r1_res_sign_1;
            r1b_res_sign_2 <= r1_res_sign_2; r1b_res_sign_3 <= r1_res_sign_3;
            r1b_res_sign_4 <= r1_res_sign_4; r1b_res_sign_5 <= r1_res_sign_5;
            r1b_res_sign_6 <= r1_res_sign_6; r1b_res_sign_7 <= r1_res_sign_7;
            r1b_zero_sign_0 <= r1_zero_sign_0; r1b_zero_sign_1 <= r1_zero_sign_1;
            r1b_zero_sign_2 <= r1_zero_sign_2; r1b_zero_sign_3 <= r1_zero_sign_3;
            r1b_zero_sign_4 <= r1_zero_sign_4; r1b_zero_sign_5 <= r1_zero_sign_5;
            r1b_zero_sign_6 <= r1_zero_sign_6; r1b_zero_sign_7 <= r1_zero_sign_7;
            r1b_exp_common_0 <= r1_exp_common_0; r1b_exp_common_1 <= r1_exp_common_1;
            r1b_exp_common_2 <= r1_exp_common_2; r1b_exp_common_3 <= r1_exp_common_3;
            r1b_exp_common_4 <= r1_exp_common_4; r1b_exp_common_5 <= r1_exp_common_5;
            r1b_exp_common_6 <= r1_exp_common_6; r1b_exp_common_7 <= r1_exp_common_7;
            r1b_mode <= r1_mode;
            r1b_match_0 <= r1_match_0; r1b_match_1 <= r1_match_1; r1b_match_2 <= r1_match_2; r1b_match_3 <= r1_match_3;
            r1b_match_4 <= r1_match_4; r1b_match_5 <= r1_match_5; r1b_match_6 <= r1_match_6; r1b_match_7 <= r1_match_7;
            r1b_spec_res_0 <= r1_spec_res_0; r1b_spec_res_1 <= r1_spec_res_1;
            r1b_spec_res_2 <= r1_spec_res_2; r1b_spec_res_3 <= r1_spec_res_3;
        end
    end

    // ================= STAGE 2b: mantissa add =================
    wire [M32+2:0] s2_frac_out_0, s2_frac_out_1, s2_frac_out_2, s2_frac_out_3;
    wire [M32+2:0] s2_frac_out_4, s2_frac_out_5, s2_frac_out_6, s2_frac_out_7;

    // base +/- aligned operand in the shared segmented adder, sliced per lane
    // (26-bit lane results including the overflow bit).
    add_slice_simd #( .E(E32), .M(M32), .M16(M16), .M8(M8), .M4(M4) ) u_add_slice (
        .mode_control(r1b_mode),
        .base32(r1b_base32), .shifted32(r1b_shifted32),
        .signed_xor_0(r1b_signed_xor_0), .signed_xor_1(r1b_signed_xor_1), .signed_xor_2(r1b_signed_xor_2), .signed_xor_3(r1b_signed_xor_3),
        .signed_xor_4(r1b_signed_xor_4), .signed_xor_5(r1b_signed_xor_5), .signed_xor_6(r1b_signed_xor_6), .signed_xor_7(r1b_signed_xor_7),
        .frac_out_0(s2_frac_out_0), .frac_out_1(s2_frac_out_1), .frac_out_2(s2_frac_out_2), .frac_out_3(s2_frac_out_3),
        .frac_out_4(s2_frac_out_4), .frac_out_5(s2_frac_out_5), .frac_out_6(s2_frac_out_6), .frac_out_7(s2_frac_out_7)
    );

    // ================= REGISTER 2 =================
    (* shreg_extract = "no" *) reg [M32+2:0] r2_frac_out_0, r2_frac_out_1, r2_frac_out_2, r2_frac_out_3;
    (* shreg_extract = "no" *) reg [M32+2:0] r2_frac_out_4, r2_frac_out_5, r2_frac_out_6, r2_frac_out_7;
    (* shreg_extract = "no" *) reg r2_res_sign_0, r2_res_sign_1, r2_res_sign_2, r2_res_sign_3;
    (* shreg_extract = "no" *) reg r2_res_sign_4, r2_res_sign_5, r2_res_sign_6, r2_res_sign_7;
    (* shreg_extract = "no" *) reg r2_zero_sign_0, r2_zero_sign_1, r2_zero_sign_2, r2_zero_sign_3;
    (* shreg_extract = "no" *) reg r2_zero_sign_4, r2_zero_sign_5, r2_zero_sign_6, r2_zero_sign_7;
    (* shreg_extract = "no" *) reg [E32-1:0] r2_exp_common_0, r2_exp_common_1, r2_exp_common_2, r2_exp_common_3;
    (* shreg_extract = "no" *) reg [E32-1:0] r2_exp_common_4, r2_exp_common_5, r2_exp_common_6, r2_exp_common_7;
    (* shreg_extract = "no" *) reg [1:0] r2_mode;
    (* shreg_extract = "no" *) reg r2_match_0, r2_match_1, r2_match_2, r2_match_3, r2_match_4, r2_match_5, r2_match_6, r2_match_7;
    (* shreg_extract = "no" *) reg [31:0] r2_spec_res_0;
    (* shreg_extract = "no" *) reg [15:0] r2_spec_res_1;
    (* shreg_extract = "no" *) reg [7:0]  r2_spec_res_2, r2_spec_res_3;

    always @(posedge clk) begin
        if (ce) begin
            r2_frac_out_0 <= s2_frac_out_0; r2_frac_out_1 <= s2_frac_out_1;
            r2_frac_out_2 <= s2_frac_out_2; r2_frac_out_3 <= s2_frac_out_3;
            r2_frac_out_4 <= s2_frac_out_4; r2_frac_out_5 <= s2_frac_out_5;
            r2_frac_out_6 <= s2_frac_out_6; r2_frac_out_7 <= s2_frac_out_7;
            r2_res_sign_0 <= r1b_res_sign_0; r2_res_sign_1 <= r1b_res_sign_1;
            r2_res_sign_2 <= r1b_res_sign_2; r2_res_sign_3 <= r1b_res_sign_3;
            r2_res_sign_4 <= r1b_res_sign_4; r2_res_sign_5 <= r1b_res_sign_5;
            r2_res_sign_6 <= r1b_res_sign_6; r2_res_sign_7 <= r1b_res_sign_7;
            r2_zero_sign_0 <= r1b_zero_sign_0; r2_zero_sign_1 <= r1b_zero_sign_1;
            r2_zero_sign_2 <= r1b_zero_sign_2; r2_zero_sign_3 <= r1b_zero_sign_3;
            r2_zero_sign_4 <= r1b_zero_sign_4; r2_zero_sign_5 <= r1b_zero_sign_5;
            r2_zero_sign_6 <= r1b_zero_sign_6; r2_zero_sign_7 <= r1b_zero_sign_7;
            r2_exp_common_0 <= r1b_exp_common_0; r2_exp_common_1 <= r1b_exp_common_1;
            r2_exp_common_2 <= r1b_exp_common_2; r2_exp_common_3 <= r1b_exp_common_3;
            r2_exp_common_4 <= r1b_exp_common_4; r2_exp_common_5 <= r1b_exp_common_5;
            r2_exp_common_6 <= r1b_exp_common_6; r2_exp_common_7 <= r1b_exp_common_7;
            r2_mode <= r1b_mode;
            r2_match_0 <= r1b_match_0; r2_match_1 <= r1b_match_1; r2_match_2 <= r1b_match_2; r2_match_3 <= r1b_match_3;
            r2_match_4 <= r1b_match_4; r2_match_5 <= r1b_match_5; r2_match_6 <= r1b_match_6; r2_match_7 <= r1b_match_7;
            r2_spec_res_0 <= r1b_spec_res_0; r2_spec_res_1 <= r1b_spec_res_1;
            r2_spec_res_2 <= r1b_spec_res_2; r2_spec_res_3 <= r1b_spec_res_3;
        end
    end

    // ================= REGISTER 2b (sideband delay for the Stage 3 cut) =================
    // Delays the sideband signals (signs, mode, special-case results) by one
    // cycle so they stay aligned with pack_decide_simd's internal register stage.
    (* shreg_extract = "no" *) reg r2b_res_sign_0, r2b_res_sign_1, r2b_res_sign_2, r2b_res_sign_3, r2b_res_sign_4, r2b_res_sign_5, r2b_res_sign_6, r2b_res_sign_7;
    (* shreg_extract = "no" *) reg r2b_zero_sign_0, r2b_zero_sign_1, r2b_zero_sign_2, r2b_zero_sign_3, r2b_zero_sign_4, r2b_zero_sign_5, r2b_zero_sign_6, r2b_zero_sign_7;
    (* shreg_extract = "no" *) reg [1:0] r2b_mode;
    (* shreg_extract = "no" *) reg r2b_match_0, r2b_match_1, r2b_match_2, r2b_match_3, r2b_match_4, r2b_match_5, r2b_match_6, r2b_match_7;
    (* shreg_extract = "no" *) reg [31:0] r2b_spec_res_0;
    (* shreg_extract = "no" *) reg [15:0] r2b_spec_res_1;
    (* shreg_extract = "no" *) reg [7:0]  r2b_spec_res_2, r2b_spec_res_3;
    always @(posedge clk) if (ce) begin
        r2b_res_sign_0 <= r2_res_sign_0; r2b_zero_sign_0 <= r2_zero_sign_0; r2b_match_0 <= r2_match_0;
        r2b_res_sign_1 <= r2_res_sign_1; r2b_zero_sign_1 <= r2_zero_sign_1; r2b_match_1 <= r2_match_1;
        r2b_res_sign_2 <= r2_res_sign_2; r2b_zero_sign_2 <= r2_zero_sign_2; r2b_match_2 <= r2_match_2;
        r2b_res_sign_3 <= r2_res_sign_3; r2b_zero_sign_3 <= r2_zero_sign_3; r2b_match_3 <= r2_match_3;
        r2b_res_sign_4 <= r2_res_sign_4; r2b_zero_sign_4 <= r2_zero_sign_4; r2b_match_4 <= r2_match_4;
        r2b_res_sign_5 <= r2_res_sign_5; r2b_zero_sign_5 <= r2_zero_sign_5; r2b_match_5 <= r2_match_5;
        r2b_res_sign_6 <= r2_res_sign_6; r2b_zero_sign_6 <= r2_zero_sign_6; r2b_match_6 <= r2_match_6;
        r2b_res_sign_7 <= r2_res_sign_7; r2b_zero_sign_7 <= r2_zero_sign_7; r2b_match_7 <= r2_match_7;
        r2b_mode <= r2_mode;
        r2b_spec_res_0 <= r2_spec_res_0; r2b_spec_res_1 <= r2_spec_res_1; r2b_spec_res_2 <= r2_spec_res_2; r2b_spec_res_3 <= r2_spec_res_3;
    end

    // ================= STAGE 3 =================
    // Stage 3 decisions from pack_decide_simd: is_zero, shift_src, act_shift and
    // final_exp per lane, plus flush-to-zero flags for lanes 0 and 1.
    wire s3_is_zero_0, s3_is_zero_1, s3_is_zero_2, s3_is_zero_3, s3_is_zero_4, s3_is_zero_5, s3_is_zero_6, s3_is_zero_7;
    wire [M32+1:0] s3_shift_src_0;
    wire [M16+1:0] s3_shift_src_1;
    wire [M8+1:0]  s3_shift_src_2, s3_shift_src_3;
    wire [M4+1:0]  s3_shift_src_4, s3_shift_src_5, s3_shift_src_6, s3_shift_src_7;
    wire [4:0] s3_act_shift_0;
    wire [3:0] s3_act_shift_1;
    wire [2:0] s3_act_shift_2, s3_act_shift_3;
    wire [1:0] s3_act_shift_4, s3_act_shift_5, s3_act_shift_6, s3_act_shift_7;
    wire [E32:0] s3_final_exp_0, s3_final_exp_1, s3_final_exp_2, s3_final_exp_3;
    wire [E32:0] s3_final_exp_4, s3_final_exp_5, s3_final_exp_6, s3_final_exp_7;
    wire s3_ftz_0, s3_ftz_1;

    // Leading-zero count, normalization shift amount and final exponent. The
    // module has an internal register stage, so its outputs are one cycle later
    // than r2_*.
    pack_decide_simd #( .FTZ_EN(FTZ_EN), .E32(E32), .M32(M32), .E16(E16), .M16(M16), .E8(E8), .M8(M8), .E4(E4), .M4(M4) ) u_pack_decide (
        .clk(clk), .ce(ce),
        .mode_control(r2_mode),
        .mant_res_0(r2_frac_out_0), .mant_res_1(r2_frac_out_1), .mant_res_2(r2_frac_out_2), .mant_res_3(r2_frac_out_3),
        .mant_res_4(r2_frac_out_4), .mant_res_5(r2_frac_out_5), .mant_res_6(r2_frac_out_6), .mant_res_7(r2_frac_out_7),
        .exp_common_0(r2_exp_common_0), .exp_common_1(r2_exp_common_1), .exp_common_2(r2_exp_common_2), .exp_common_3(r2_exp_common_3),
        .exp_common_4(r2_exp_common_4), .exp_common_5(r2_exp_common_5), .exp_common_6(r2_exp_common_6), .exp_common_7(r2_exp_common_7),
        .is_zero_0(s3_is_zero_0), .shift_src_0(s3_shift_src_0), .act_shift_0(s3_act_shift_0), .final_exp_0(s3_final_exp_0),
        .is_zero_1(s3_is_zero_1), .shift_src_1(s3_shift_src_1), .act_shift_1(s3_act_shift_1), .final_exp_1(s3_final_exp_1),
        .is_zero_2(s3_is_zero_2), .shift_src_2(s3_shift_src_2), .act_shift_2(s3_act_shift_2), .final_exp_2(s3_final_exp_2),
        .is_zero_3(s3_is_zero_3), .shift_src_3(s3_shift_src_3), .act_shift_3(s3_act_shift_3), .final_exp_3(s3_final_exp_3),
        .is_zero_4(s3_is_zero_4), .shift_src_4(s3_shift_src_4), .act_shift_4(s3_act_shift_4), .final_exp_4(s3_final_exp_4),
        .is_zero_5(s3_is_zero_5), .shift_src_5(s3_shift_src_5), .act_shift_5(s3_act_shift_5), .final_exp_5(s3_final_exp_5),
        .is_zero_6(s3_is_zero_6), .shift_src_6(s3_shift_src_6), .act_shift_6(s3_act_shift_6), .final_exp_6(s3_final_exp_6),
        .is_zero_7(s3_is_zero_7), .shift_src_7(s3_shift_src_7), .act_shift_7(s3_act_shift_7), .final_exp_7(s3_final_exp_7),
        .ftz_0(s3_ftz_0), .ftz_1(s3_ftz_1)
    );

    // ================= REGISTER 3 =================
    (* shreg_extract = "no" *) reg r3_is_zero_0, r3_is_zero_1, r3_is_zero_2, r3_is_zero_3, r3_is_zero_4, r3_is_zero_5, r3_is_zero_6, r3_is_zero_7;
    (* shreg_extract = "no" *) reg [M32+1:0] r3_shift_src_0;
    (* shreg_extract = "no" *) reg [M16+1:0] r3_shift_src_1;
    (* shreg_extract = "no" *) reg [M8+1:0]  r3_shift_src_2, r3_shift_src_3;
    (* shreg_extract = "no" *) reg [M4+1:0]  r3_shift_src_4, r3_shift_src_5, r3_shift_src_6, r3_shift_src_7;
    (* shreg_extract = "no" *) reg [4:0] r3_act_shift_0;
    (* shreg_extract = "no" *) reg [3:0] r3_act_shift_1;
    (* shreg_extract = "no" *) reg [2:0] r3_act_shift_2, r3_act_shift_3;
    (* shreg_extract = "no" *) reg [1:0] r3_act_shift_4, r3_act_shift_5, r3_act_shift_6, r3_act_shift_7;
    (* shreg_extract = "no" *) reg [E32:0] r3_final_exp_0, r3_final_exp_1, r3_final_exp_2, r3_final_exp_3;
    (* shreg_extract = "no" *) reg [E32:0] r3_final_exp_4, r3_final_exp_5, r3_final_exp_6, r3_final_exp_7;
    (* shreg_extract = "no" *) reg r3_res_sign_0, r3_res_sign_1, r3_res_sign_2, r3_res_sign_3;
    (* shreg_extract = "no" *) reg r3_res_sign_4, r3_res_sign_5, r3_res_sign_6, r3_res_sign_7;
    (* shreg_extract = "no" *) reg r3_zero_sign_0, r3_zero_sign_1, r3_zero_sign_2, r3_zero_sign_3;
    (* shreg_extract = "no" *) reg r3_zero_sign_4, r3_zero_sign_5, r3_zero_sign_6, r3_zero_sign_7;
    (* shreg_extract = "no" *) reg [1:0] r3_mode;
    (* shreg_extract = "no" *) reg r3_match_0, r3_match_1, r3_match_2, r3_match_3, r3_match_4, r3_match_5, r3_match_6, r3_match_7;
    (* shreg_extract = "no" *) reg [31:0] r3_spec_res_0;
    (* shreg_extract = "no" *) reg [15:0] r3_spec_res_1;
    (* shreg_extract = "no" *) reg [7:0]  r3_spec_res_2, r3_spec_res_3;

    always @(posedge clk) begin
        if (ce) begin
            // A flushed (subnormal) FP32/FP16 result is treated as zero here, and
            // (see r3_zero_sign_0/1 below) takes the sign of the result rather than the
            // zero-sum sign.
            r3_is_zero_0 <= s3_is_zero_0 | s3_ftz_0; r3_is_zero_1 <= s3_is_zero_1 | s3_ftz_1;
            r3_is_zero_2 <= s3_is_zero_2; r3_is_zero_3 <= s3_is_zero_3;
            r3_is_zero_4 <= s3_is_zero_4; r3_is_zero_5 <= s3_is_zero_5;
            r3_is_zero_6 <= s3_is_zero_6; r3_is_zero_7 <= s3_is_zero_7;
            r3_shift_src_0 <= s3_shift_src_0; r3_shift_src_1 <= s3_shift_src_1;
            r3_shift_src_2 <= s3_shift_src_2; r3_shift_src_3 <= s3_shift_src_3;
            r3_shift_src_4 <= s3_shift_src_4; r3_shift_src_5 <= s3_shift_src_5;
            r3_shift_src_6 <= s3_shift_src_6; r3_shift_src_7 <= s3_shift_src_7;
            r3_act_shift_0 <= s3_act_shift_0; r3_act_shift_1 <= s3_act_shift_1;
            r3_act_shift_2 <= s3_act_shift_2; r3_act_shift_3 <= s3_act_shift_3;
            r3_act_shift_4 <= s3_act_shift_4; r3_act_shift_5 <= s3_act_shift_5;
            r3_act_shift_6 <= s3_act_shift_6; r3_act_shift_7 <= s3_act_shift_7;
            r3_final_exp_0 <= s3_final_exp_0; r3_final_exp_1 <= s3_final_exp_1;
            r3_final_exp_2 <= s3_final_exp_2; r3_final_exp_3 <= s3_final_exp_3;
            r3_final_exp_4 <= s3_final_exp_4; r3_final_exp_5 <= s3_final_exp_5;
            r3_final_exp_6 <= s3_final_exp_6; r3_final_exp_7 <= s3_final_exp_7;
            r3_res_sign_0 <= r2b_res_sign_0; r3_res_sign_1 <= r2b_res_sign_1;
            r3_res_sign_2 <= r2b_res_sign_2; r3_res_sign_3 <= r2b_res_sign_3;
            r3_res_sign_4 <= r2b_res_sign_4; r3_res_sign_5 <= r2b_res_sign_5;
            r3_res_sign_6 <= r2b_res_sign_6; r3_res_sign_7 <= r2b_res_sign_7;
            r3_zero_sign_0 <= s3_ftz_0 ? r2b_res_sign_0 : r2b_zero_sign_0; r3_zero_sign_1 <= s3_ftz_1 ? r2b_res_sign_1 : r2b_zero_sign_1;
            r3_zero_sign_2 <= r2b_zero_sign_2; r3_zero_sign_3 <= r2b_zero_sign_3;
            r3_zero_sign_4 <= r2b_zero_sign_4; r3_zero_sign_5 <= r2b_zero_sign_5;
            r3_zero_sign_6 <= r2b_zero_sign_6; r3_zero_sign_7 <= r2b_zero_sign_7;
            r3_mode <= r2b_mode;
            r3_match_0 <= r2b_match_0; r3_match_1 <= r2b_match_1; r3_match_2 <= r2b_match_2; r3_match_3 <= r2b_match_3;
            r3_match_4 <= r2b_match_4; r3_match_5 <= r2b_match_5; r3_match_6 <= r2b_match_6; r3_match_7 <= r2b_match_7;
            r3_spec_res_0 <= r2b_spec_res_0; r3_spec_res_1 <= r2b_spec_res_1;
            r3_spec_res_2 <= r2b_spec_res_2; r3_spec_res_3 <= r2b_spec_res_3;
        end
    end

    // ================= STAGE 4 =================
    wire [31:0] pack_result;

    // Apply the normalization shift and encode each format's result word.
    pack_execute_simd #( .E32(E32), .M32(M32), .E16(E16), .M16(M16), .E8(E8), .M8(M8), .E4(E4), .M4(M4) ) u_pack_execute (
        .mode_control(r3_mode),
        .is_zero_0(r3_is_zero_0), .shift_src_0(r3_shift_src_0), .act_shift_0(r3_act_shift_0), .final_exp_0(r3_final_exp_0),
        .is_zero_1(r3_is_zero_1), .shift_src_1(r3_shift_src_1), .act_shift_1(r3_act_shift_1), .final_exp_1(r3_final_exp_1),
        .is_zero_2(r3_is_zero_2), .shift_src_2(r3_shift_src_2), .act_shift_2(r3_act_shift_2), .final_exp_2(r3_final_exp_2),
        .is_zero_3(r3_is_zero_3), .shift_src_3(r3_shift_src_3), .act_shift_3(r3_act_shift_3), .final_exp_3(r3_final_exp_3),
        .is_zero_4(r3_is_zero_4), .shift_src_4(r3_shift_src_4), .act_shift_4(r3_act_shift_4), .final_exp_4(r3_final_exp_4),
        .is_zero_5(r3_is_zero_5), .shift_src_5(r3_shift_src_5), .act_shift_5(r3_act_shift_5), .final_exp_5(r3_final_exp_5),
        .is_zero_6(r3_is_zero_6), .shift_src_6(r3_shift_src_6), .act_shift_6(r3_act_shift_6), .final_exp_6(r3_final_exp_6),
        .is_zero_7(r3_is_zero_7), .shift_src_7(r3_shift_src_7), .act_shift_7(r3_act_shift_7), .final_exp_7(r3_final_exp_7),
        .res_sign_0(r3_res_sign_0), .res_sign_1(r3_res_sign_1), .res_sign_2(r3_res_sign_2), .res_sign_3(r3_res_sign_3),
        .res_sign_4(r3_res_sign_4), .res_sign_5(r3_res_sign_5), .res_sign_6(r3_res_sign_6), .res_sign_7(r3_res_sign_7),
        .zero_sign_0(r3_zero_sign_0), .zero_sign_1(r3_zero_sign_1), .zero_sign_2(r3_zero_sign_2), .zero_sign_3(r3_zero_sign_3),
        .zero_sign_4(r3_zero_sign_4), .zero_sign_5(r3_zero_sign_5), .zero_sign_6(r3_zero_sign_6), .zero_sign_7(r3_zero_sign_7),
        .result(pack_result)
    );

    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;  // mode_control encodings

    // Special-case bypass: when a lane's match flag is set, the Inf/NaN value
    // from stage 1 replaces that lane's arithmetic result. Each format below
    // views pack_result in its own lane width.
    // 8-bit-lane view (used by FP8 mode)
    wire [7:0]  final_lane_0 = r3_match_0 ? r3_spec_res_0[7:0] : pack_result[7:0];
    wire [7:0]  final_lane_1 = r3_match_1 ? r3_spec_res_1[7:0] : pack_result[15:8];
    wire [7:0]  final_lane_2 = r3_match_2 ? r3_spec_res_2      : pack_result[23:16];
    wire [7:0]  final_lane_3 = r3_match_3 ? r3_spec_res_3      : pack_result[31:24];

    // 16-bit-lane view (used by FP16 mode)
    wire [15:0] final_fp16_0 = r3_match_0 ? r3_spec_res_0[15:0] : pack_result[15:0];
    wire [15:0] final_fp16_1 = r3_match_1 ? r3_spec_res_1       : pack_result[31:16];

    // 32-bit view (used by FP32 mode)
    wire [31:0] final_fp32   = r3_match_0 ? r3_spec_res_0 : pack_result;

    // 4-bit-lane view (used by FP4 mode) -- no special-case bypass for FP4
    wire [31:0] final_fp4 = pack_result;

    // Select the lane view for the active format.
    assign result = (r3_mode == FP4)  ? final_fp4 :
                     (r3_mode == FP8)  ? {final_lane_3, final_lane_2, final_lane_1, final_lane_0} :
                     (r3_mode == FP16) ? {final_fp16_1, final_fp16_0} :
                                          final_fp32;

endmodule