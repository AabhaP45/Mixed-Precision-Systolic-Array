// =========================================================================
// fane_mac_simd -- shared-datapath MAC unit (FP32 / 2xFP16 / 4xFP8 / 8xFP4)
//
// Updated for the 5-stage addmul_simd (E4M3FN) and 5-stage fp_adder_simd.
//
// acc_out = (mul_a * mul_b) + cascade_sum_in, per SIMD lane.
//
// Timing (cycle 0 = cycle mul_a/mul_b/cascade_sum_in/mode_control are applied):
//   cycle 1 : acin_reg / bcin_reg / cascade_sum_in_reg / mode_reg0 valid
//   cycle 6 : addmul result (fp_product) valid   (addmul latency = 5)
//   cycle 11: adder  result (sum_result) valid   (adder  latency = 5)
//   cycle 12: acc_out valid                      (output register)
// The adder operand `a` (cascade sum) and its mode_control must therefore be
// delayed by ADDMUL_LAT extra cycles so they arrive together with fp_product.
// =========================================================================
module fane_mac_simd(
    input               clk,
    input               rst_n,     // Asynchronous, active-low reset
    input               ce,        // Global clock enable
    input               ce_a_1,    // Clock enable for mul_a register stage
    input               ce_a_2,    // (Unused here but kept for pipeline control)
    input               ce_b_1,    // Clock enable for mul_b register stage
    input               ce_b_2,    // (Unused here but kept for pipeline control)

    input   [1:0]       mode_control,      // Selects the SIMD precision/lane format

    input   [31:0]      mul_a,             // Multiplicand
    input   [31:0]      mul_b,             // Multiplier
    input   [31:0]      cascade_sum_in,    // Cascaded partial sum (PCIN)

    output  [31:0]      acc_out,           // Final accumulated output (P)
    output  [31:0]      cascade_mula_out,  // Cascaded output of multiplicand
    output  [31:0]      cascade_mulb_out   // Cascaded output of multiplier
);

// Pipeline depth parameters; TOTAL_LAT counts the input register, both
// arithmetic units and the output register.
localparam ADDMUL_LAT = 5;                    // addmul_simd: input reg + 4 stages
localparam ADDER_LAT  = 5;                    // fp_adder_simd: r1,r1b,r2,r2b,r3
localparam TOTAL_LAT  = 1 + ADDMUL_LAT + ADDER_LAT + 1;  // = 12

// Internal signals
wire [31:0] fp_product;             // Multiplier result
wire [31:0] sum_result;             // Adder result (product + aligned cascade sum)
reg  [31:0] acin_reg;               // Registered multiplicand
reg  [31:0] bcin_reg;               // Registered multiplier
reg  [1:0]  mode_reg0;              // Registered mode for the multiplier stage
reg  [31:0] cascade_sum_in_reg;     // Registered cascade sum input
reg  [31:0] cascade_mula_out_reg;   // Registered multiplicand pass-through
reg  [31:0] cascade_mulb_out_reg;   // Registered multiplier pass-through
reg  [31:0] acc_out_reg;            // Output accumulator register

// Delay line that carries the cascade sum and mode alongside the multiplier
(* dont_touch = "true" *) reg  [31:0] casc_dly [0:ADDMUL_LAT-1];
(* dont_touch = "true" *) reg  [1:0]  mode_dly [0:ADDMUL_LAT-1];
integer i;                          // Loop index for the delay line

// Pipeline valid shift register to mask power-on X's
reg  [TOTAL_LAT-1:0] pipe_valid;

// Valid tracker: MSB goes high once the first input reaches the output
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pipe_valid <= {TOTAL_LAT{1'b0}};
    end else if (ce) begin
        pipe_valid <= {pipe_valid[TOTAL_LAT-2:0], 1'b1};
    end
end

// Input stage for the cascade sum and mode control
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cascade_sum_in_reg <= 32'd0;
        mode_reg0          <= 2'd0;
    end else if (ce) begin
        cascade_sum_in_reg <= cascade_sum_in;
        mode_reg0          <= mode_control;
    end
end

// Input register for the multiplicand (gated by its own enable)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) acin_reg <= 32'd0;
    else if (ce_a_1) acin_reg <= mul_a;
end

// Input register for the multiplier (gated by its own enable)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) bcin_reg <= 32'd0;
    else if (ce_b_1) bcin_reg <= mul_b;
end

// Operand pass-through registers for chaining to the next MAC in a cascade
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cascade_mula_out_reg <= 32'd0;
        cascade_mulb_out_reg <= 32'd0;
    end else if (ce) begin
        cascade_mula_out_reg <= mul_a;
        cascade_mulb_out_reg <= mul_b;
    end
end

assign cascade_mula_out = cascade_mula_out_reg;
assign cascade_mulb_out = cascade_mulb_out_reg;

// ---------------------------------------------------------------- multiplier
(* dont_touch = "true" *) addmul_simd U_FP_ADDMUL(
    .clk          (clk        ),
    .ce           (ce         ),
    .mode_control (mode_reg0  ),
    .a            (acin_reg   ),
    .b            (bcin_reg   ),
    .result       (fp_product )
);

// ------------------------------------------- align cascade sum / mode (5 cyc)
// Delay cascade sum/mode to match multiplier latency
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < ADDMUL_LAT; i = i + 1) begin
            casc_dly[i] <= 32'd0;
            mode_dly[i] <= 2'd0;
        end
    end else if (ce) begin
        casc_dly[0] <= cascade_sum_in_reg;
        mode_dly[0] <= mode_reg0;
        for (i = 1; i < ADDMUL_LAT; i = i + 1) begin
            casc_dly[i] <= casc_dly[i-1];
            mode_dly[i] <= mode_dly[i-1];
        end
    end
end

// ------------------------------------------------------------------- adder
// Adds the delayed cascade sum (a) to the multiplier product (b)
(* dont_touch = "true" *) fp_adder_simd U_FP_ADDER(
    .clk          (clk                      ),
    .ce           (ce                       ),
    .mode_control (mode_dly[ADDMUL_LAT-1]   ),
    .a            (casc_dly[ADDMUL_LAT-1]   ),
    .b            (fp_product               ),
    .result       (sum_result               )
);

// Output register: mask the X's using pipe_valid
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc_out_reg <= 32'd0;
    end else if (ce) begin
        if (pipe_valid[TOTAL_LAT-1])
            acc_out_reg <= sum_result;
        else
            acc_out_reg <= 32'd0;
    end
end

assign acc_out = acc_out_reg;

endmodule