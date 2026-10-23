// -----------------------------------------------------------------------------
// addmul_simd.v
//
// 5-stage SIMD Mitchell approximate multiplier (log-domain field adder).
// Supports cycle-by-cycle mode switching across 32-bit packed lanes:
//   mode_control = 2'b00 : 1x FP32       (1S, 8E, 23M, bias 127)
//   mode_control = 2'b01 : 2x FP16       (1S, 5E, 10M, bias 15)
//   mode_control = 2'b10 : 4x FP8 E4M3FN (1S, 4E, 3M,  bias 7)
//   mode_control = 2'b11 : 8x FP4 E2M1   (1S, 2E, 1M,  bias 1)
//
// Pipeline overview (5 cycles total, gated by `ce`):
//   P  : Register input operands and mode_control.
//   S1 : Mask lane sign bits and add {exp, frac} fields in a single 32-bit adder.
//        Evaluate nibble-level zero, Inf, and NaN flags.
//   S2 : Subtract exponent bias by adding 2^(w-1) - bias. The carry into the
//        lane MSB indicates (sum1 >= bias), avoiding separate comparators.
//   S3 : Compute per-lane kill, saturation (e1), sign (sg), and NaN (nb) flags.
//   S4 : Broadcast flags from sign nibbles across lane nibbles and mux final bits.
// -----------------------------------------------------------------------------
module addmul_simd (
    input  wire        clk,
    input  wire        ce,
    input  wire [1:0]  mode_control,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] result
);
    localparam FP32 = 2'b00, FP16 = 2'b01, FP8 = 2'b10, FP4 = 2'b11;

    // --- Stage P: Input registers ---
    reg [1:0]  pipe_mode;
    reg [31:0] pipe_a, pipe_b;
    always @(posedge clk) if (ce) begin
        pipe_mode <= mode_control;
        pipe_a    <= a;
        pipe_b    <= b;
    end

    // --- Stage S1: Field addition & special-case detect ---
    wire m32 = (pipe_mode == FP32);
    wire m16 = (pipe_mode == FP16);
    wire m8  = (pipe_mode == FP8);
    wire m4  = (pipe_mode == FP4);

    // Mark nibbles whose MSB (bit 4n+3) acts as a lane sign bit in the current mode
    wire [7:0]  top     = m4 ? 8'hFF : m8 ? 8'hAA : m16 ? 8'h88 : 8'h80;
    wire [31:0] topmask = {top[7],3'b0, top[6],3'b0, top[5],3'b0, top[4],3'b0,
                           top[3],3'b0, top[2],3'b0, top[1],3'b0, top[0],3'b0};

    // Clear lane sign bits before adding so carries never propagate across lanes
    wire [31:0] s1_sum1 = (pipe_a & ~topmask) + (pipe_b & ~topmask);

    // Nibble reduction flags for zero and all-ones checks:
    //   zl[n]: bits [4n+2:4n] == 0   | zf[n]: bits [4n+3:4n] == 0
    //   oe[k]: bits [8k+3:8k] == 4'hF| oo[k]: bits [8k+6:8k+4] == 3'b111
    wire [7:0] zl_a, zl_b;
    wire [6:0] zf_a, zf_b;
    wire [3:0] oe_a, oe_b, oo_a, oo_b;
    genvar n;
    generate
        for (n = 0; n < 8; n = n + 1) begin : g_zl
            assign zl_a[n] = ~|pipe_a[4*n +: 3];
            assign zl_b[n] = ~|pipe_b[4*n +: 3];
        end
        for (n = 0; n < 7; n = n + 1) begin : g_zf
            assign zf_a[n] = ~|pipe_a[4*n +: 4];
            assign zf_b[n] = ~|pipe_b[4*n +: 4];
        end
        for (n = 0; n < 4; n = n + 1) begin : g_ones
            assign oe_a[n] = &pipe_a[8*n +: 4];      assign oe_b[n] = &pipe_b[8*n +: 4];
            assign oo_a[n] = &pipe_a[8*n+4 +: 3];    assign oo_b[n] = &pipe_b[8*n+4 +: 3];
        end
    endgenerate

    // Check if the lane ending at sign-nibble `t` is zero (or subnormal flushed to zero)
    function [7:0] zero_tops(input [7:0] zl, input [6:0] zf, input mm4, input mm8, input mm16);
        begin
            zero_tops[0] = zl[0];
            zero_tops[1] = zl[1] & (mm4 | zf[0]);
            zero_tops[2] = zl[2];
            zero_tops[3] = zl[3] & (mm4 | (zf[2] & (mm8 | (zf[0] & zf[1]))));
            zero_tops[4] = zl[4];
            zero_tops[5] = zl[5] & (mm4 | zf[4]);
            zero_tops[6] = zl[6];
            zero_tops[7] = zl[7] & (mm4 | (zf[6] & (mm8 | (zf[4] & zf[5] & (mm16 | (zf[0] & zf[1] & zf[2] & zf[3]))))));
        end
    endfunction
    wire [7:0] s1_z = zero_tops(zl_a, zf_a, m4, m8, m16) | zero_tops(zl_b, zf_b, m4, m8, m16);

    // Product sign at each nibble MSB
    wire [31:0] sxor = pipe_a ^ pipe_b;
    wire [7:0]  s1_sx = {sxor[31], sxor[27], sxor[23], sxor[19], sxor[15], sxor[11], sxor[7], sxor[3]};

    // Inf / NaN detection for FP32 (lane 0) and FP16 (lanes 0, 1)
    wire [1:0] hz_a = {~|pipe_a[25:24], ~|pipe_a[9:8]};
    wire [1:0] hz_b = {~|pipe_b[25:24], ~|pipe_b[9:8]};
    wire e32_a  = oo_a[3] & oe_a[3] & pipe_a[23];
    wire e32_b  = oo_b[3] & oe_b[3] & pipe_b[23];
    wire e16l_a = oo_a[1] & pipe_a[11] & pipe_a[10];
    wire e16l_b = oo_b[1] & pipe_b[11] & pipe_b[10];
    wire e16h_a = oo_a[3] & pipe_a[27] & pipe_a[26];
    wire e16h_b = oo_b[3] & pipe_b[27] & pipe_b[26];
    wire f32_a = ~(zf_a[0] & zf_a[1] & zf_a[2] & zf_a[3] & zf_a[4] & zl_a[5]);
    wire f32_b = ~(zf_b[0] & zf_b[1] & zf_b[2] & zf_b[3] & zf_b[4] & zl_b[5]);
    wire f16l_a = ~(zf_a[0] & zf_a[1] & hz_a[0]);
    wire f16l_b = ~(zf_b[0] & zf_b[1] & hz_b[0]);
    wire f16h_a = ~(zf_a[4] & zf_a[5] & hz_a[1]);
    wire f16h_b = ~(zf_b[4] & zf_b[5] & hz_b[1]);

    // e0/e1 = exponent all-ones; n0/n1 = NaN (exp all-ones and frac != 0)
    wire s1_e0a = m32 ? e32_a : e16l_a;
    wire s1_e0b = m32 ? e32_b : e16l_b;
    wire s1_n0a = m32 ? (e32_a & f32_a) : (e16l_a & f16l_a);
    wire s1_n0b = m32 ? (e32_b & f32_b) : (e16l_b & f16l_b);
    wire s1_e1a = e16h_a,           s1_e1b = e16h_b;
    wire s1_n1a = e16h_a & f16h_a,  s1_n1b = e16h_b & f16h_b;
    // FP8 E4M3FN NaN is 7'h7F (exp=1111, frac=111; no Inf in E4M3FN)
    wire [3:0] s1_f8 = (oo_a & oe_a) | (oo_b & oe_b);

    // --- Pipeline register R1 (keep FFs separate from SRLs for retiming) ---
    (* shreg_extract = "no" *) reg [1:0]  r1_mode;
    (* shreg_extract = "no" *) reg [31:0] r1_sum1;
    (* shreg_extract = "no" *) reg [7:0]  r1_z, r1_sx;
    (* shreg_extract = "no" *) reg        r1_e0a, r1_e0b, r1_n0a, r1_n0b, r1_e1a, r1_e1b, r1_n1a, r1_n1b;
    (* shreg_extract = "no" *) reg [3:0]  r1_f8;
    always @(posedge clk) if (ce) begin
        r1_mode <= pipe_mode;   r1_sum1 <= s1_sum1;
        r1_z    <= s1_z;        r1_sx   <= s1_sx;
        r1_e0a <= s1_e0a; r1_e0b <= s1_e0b; r1_n0a <= s1_n0a; r1_n0b <= s1_n0b;
        r1_e1a <= s1_e1a; r1_e1b <= s1_e1b; r1_n1a <= s1_n1a; r1_n1b <= s1_n1b;
        r1_f8  <= s1_f8;
    end

    // --- Stage S2: Bias subtraction & special-case combine ---
    wire q32 = (r1_mode == FP32);
    wire q16 = (r1_mode == FP16);
    wire q8  = (r1_mode == FP8);
    wire q4  = (r1_mode == FP4);
    wire [7:0]  top2     = q4 ? 8'hFF : q8 ? 8'hAA : q16 ? 8'h88 : 8'h80;
    wire [31:0] topmask2 = {top2[7],3'b0, top2[6],3'b0, top2[5],3'b0, top2[4],3'b0,
                            top2[3],3'b0, top2[2],3'b0, top2[1],3'b0, top2[0],3'b0};

    // Subtract bias offset (FP32: 0x3F800000, FP16: 0x3C00, FP8: 0x38, FP4: 0x2)
    // by adding C = 2^(w-1) - offset. Carry into the masked lane MSB equals (sum1_low >= offset).
    wire [31:0] c2 = q32 ? 32'h4080_0000 :
                     q16 ? 32'h4400_4400 :
                     q8  ? 32'h4848_4848 :
                           32'h6666_6666;
    wire [31:0] s2_sum2 = (r1_sum1 & ~topmask2) + (c2 & ~topmask2);

    // Combine special cases: NaN if either input is NaN, or on Inf * 0
    wire match0 = r1_e0a | r1_e0b;
    wire match1 = r1_e1a | r1_e1b;
    wire z0lane = q32 ? r1_z[7] : r1_z[3];
    wire nb0    = match0 & (r1_n0a | r1_n0b | z0lane);     // bypass result is NaN
    wire nb1    = match1 & (r1_n1a | r1_n1b | r1_z[7]);

    // Route bypass (B) and NaN-bypass (NB) flags to odd sign nibbles
    wire s2_B1  = q8 & r1_f8[0];
    wire s2_B3  = q8 ? r1_f8[1] : q16 & match0;
    wire s2_B5  = q8 & r1_f8[2];
    wire s2_B7  = q8 ? r1_f8[3] : q16 ? match1 : q32 & match0;
    wire s2_NB3 = q8 ? r1_f8[1] : q16 & nb0;
    wire s2_NB7 = q8 ? r1_f8[3] : q16 ? nb1 : q32 & nb0;

    // --- Pipeline register R2 ---
    (* shreg_extract = "no" *) reg [1:0]  r2_mode;
    (* shreg_extract = "no" *) reg [31:0] r2_sum2;
    (* shreg_extract = "no" *) reg [7:0]  r2_sm, r2_z, r2_sx;      // sm: sum1 MSB per nibble
    (* shreg_extract = "no" *) reg        r2_B1, r2_B3, r2_B5, r2_B7, r2_NB3, r2_NB7;
    always @(posedge clk) if (ce) begin
        r2_mode <= r1_mode;   r2_sum2 <= s2_sum2;
        r2_sm   <= {r1_sum1[31], r1_sum1[27], r1_sum1[23], r1_sum1[19],
                    r1_sum1[15], r1_sum1[11], r1_sum1[7],  r1_sum1[3]};
        r2_z    <= r1_z;      r2_sx <= r1_sx;
        r2_B1 <= s2_B1; r2_B3 <= s2_B3; r2_B5 <= s2_B5; r2_B7 <= s2_B7;
        r2_NB3 <= s2_NB3; r2_NB7 <= s2_NB7;
    end

    // --- Stage S3: Underflow / overflow / kill / force flags ---
    // At each sign nibble: underflow = ~sm & ~ge, overflow = sm & ge, in-range = sm ^ ge
    wire [7:0] ge = {r2_sum2[31], r2_sum2[27], r2_sum2[23], r2_sum2[19],
                     r2_sum2[15], r2_sum2[11], r2_sum2[7],  r2_sum2[3]};
    wire [7:0] B  = {r2_B7, 1'b0, r2_B5, 1'b0, r2_B3, 1'b0, r2_B1, 1'b0};
    wire [7:0] NB = {r2_NB7, 1'b0, r2_B5, 1'b0, r2_NB3, 1'b0, r2_B1, 1'b0};
    wire [7:0] s3_kill = r2_z | B | ~(r2_sm ^ ge);   // zero | bypass | underflow | overflow
    wire [7:0] s3_e1   = B | (r2_sm & ge);           // bypass | overflow -> saturate/exp ones
    wire [7:0] s3_sg   = r2_sx & ~NB;                // force sign=0 on NaN
    wire [7:0] s3_nb   = NB;

    // --- Pipeline register R3 ---
    (* shreg_extract = "no" *) reg [1:0]  r3_mode;
    (* shreg_extract = "no" *) reg [31:0] r3_sum2;
    (* shreg_extract = "no" *) reg [7:0]  r3_kill, r3_e1, r3_sg, r3_nb;
    always @(posedge clk) if (ce) begin
        r3_mode <= r2_mode;  r3_sum2 <= r2_sum2;
        r3_kill <= s3_kill;  r3_e1 <= s3_e1;  r3_sg <= s3_sg;  r3_nb <= s3_nb;
    end

    // --- Stage S4: Nibble flag broadcast & bitwise output mux ---
    wire t32 = (r3_mode == FP32);
    wire t16 = (r3_mode == FP16);
    wire t8  = (r3_mode == FP8);
    wire t4  = (r3_mode == FP4);
    wire [7:0] top3 = t4 ? 8'hFF : t8 ? 8'hAA : t16 ? 8'h88 : 8'h80;

    // Map each nibble n (0..7) to the sign nibble of its lane
    wire [2:0] tsel [0:7];
    assign tsel[7] = 3'd7;
    assign tsel[6] = t4 ? 3'd6 : 3'd7;
    assign tsel[5] = (t4 | t8) ? 3'd5 : 3'd7;
    assign tsel[4] = t4 ? 3'd4 : t8 ? 3'd5 : 3'd7;
    assign tsel[3] = t32 ? 3'd7 : 3'd3;
    assign tsel[2] = t4 ? 3'd2 : t32 ? 3'd7 : 3'd3;
    assign tsel[1] = (t4 | t8) ? 3'd1 : t16 ? 3'd3 : 3'd7;
    assign tsel[0] = t4 ? 3'd0 : t8 ? 3'd1 : t16 ? 3'd3 : 3'd7;

    // Bit role within lane: 0=frac, 1=sign, 2=exp/saturate, 3=qNaN MSB
    function [1:0] bit_class(input [1:0] m, input integer i);
        integer p;
        begin
            case (m)
                FP32: p = i;
                FP16: p = i % 16;
                FP8:  p = i % 8;
                default: p = i % 4;
            endcase
            case (m)
                FP32:    bit_class = (p == 31) ? 2'd1 : (p >= 23) ? 2'd2 : (p == 22) ? 2'd3 : 2'd0;
                FP16:    bit_class = (p == 15) ? 2'd1 : (p >= 10) ? 2'd2 : (p == 9)  ? 2'd3 : 2'd0;
                FP8:     bit_class = (p == 7)  ? 2'd1 : (p >= 1)  ? 2'd2 : 2'd3;
                default: bit_class = (p == 3)  ? 2'd1 : 2'd2;
            endcase
        end
    endfunction
    function force_bit(input [1:0] cls, input sg, input e1, input nb);
        force_bit = (cls == 2'd1) ? sg : (cls == 2'd2) ? e1 : (cls == 2'd3) ? nb : 1'b0;
    endfunction

    wire [31:0] s4_result;
    genvar j;
    generate for (n = 0; n < 8; n = n + 1) begin : g_nibble
        wire kill_n = r3_kill[tsel[n]];
        wire e1_n   = r3_e1[tsel[n]];
        wire nb_n   = r3_nb[tsel[n]];
        for (j = 0; j < 4; j = j + 1) begin : g_bit
            localparam i = 4*n + j;
            wire keep = (j == 3) ? (~kill_n & ~top3[n]) : ~kill_n;
            wire f1   = (r3_mode == FP32) ? force_bit(bit_class(FP32, i), r3_sg[n], e1_n, nb_n) :
                        (r3_mode == FP16) ? force_bit(bit_class(FP16, i), r3_sg[n], e1_n, nb_n) :
                        (r3_mode == FP8)  ? force_bit(bit_class(FP8,  i), r3_sg[n], e1_n, nb_n) :
                                            force_bit(bit_class(FP4,  i), r3_sg[n], e1_n, nb_n);
            assign s4_result[i] = (r3_sum2[i] & keep) | f1;
        end
    end endgenerate

    always @(posedge clk) if (ce) result <= s4_result;
endmodule