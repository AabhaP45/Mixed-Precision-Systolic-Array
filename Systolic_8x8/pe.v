// =============================================================================
// pe_fane_simd -- output-stationary PE on fane_mac_simd
//
// The MAC is a 12-cycle feed-forward pipeline (mul_a/mul_b/cascade_sum_in at
// cycle t  ->  acc_out at cycle t+12), fully pipelined: it accepts one operand
// pair EVERY cycle and, from cycle 12 on, produces one result EVERY cycle.
//
//   acc_out(t+12) = in_a(t)*in_b(t) + cascade_sum_in(t)
//   cascade_sum_in(t) = acc_clr(t) ? 0 : psum(t)        (psum = acc_out)
//
// so  psum(t+12) = in_a(t)*in_b(t) + [clr ? 0 : psum(t)] .
// Sample t only ever sees samples t-12, t-24, ...: the PE holds 12 INTERLEAVED
// accumulators ("slots"), slot = t mod 12.  Every cycle, psum is the running
// sum of one slot.  Feed 12 independent output tiles round-robin (or hold a
// slot with zeros); after K rounds slot m holds sum_k A_m[k]*B_m[k].
//
//   acc_clr : high on the operands that START a new accumulation in their slot
//             (first product of the slot).  It is registered with in_a and
//             forwarded to the right (acc_clr_out) with the same 1-cycle hop as
//             out_a, so it stays aligned with the activation stream.
//
// Keep ce = ce_a_1 = ce_b_1 = 1 while computing (the MAC's operand registers
// and its cascade/delay registers must advance together).
// =============================================================================
module pe_fane_simd (
    input  wire        clk,                         // clock
    input  wire        rst_n,                       // active-low asynchronous reset
    input  wire        ce,                          // clock enable for the MAC pipeline and the clear-flag register
    input  wire        ce_a_1,                      // clock enable for the MAC activation (A) operand registers
    input  wire        ce_b_1,                      // clock enable for the MAC weight (B) operand registers
    input  wire [1:0]  mode_control,   // 00:FP32, 01:2xFP16, 10:4xFP8, 11:8xFP4

    input  wire [31:0] in_a,           // from left PE / row input
    input  wire [31:0] in_b,           // from top PE / column input
    input  wire        acc_clr,        // start new accumulation in this slot

    output wire [31:0] out_a,          // to right PE (1 cycle later)
    output wire [31:0] out_b,          // to bottom PE (1 cycle later)
    output wire        acc_clr_out,    // to right PE (1 cycle later)
    output wire [31:0] psum            // new result every cycle from cycle 12
);

    // Cascade (addend) input of the MAC. When acc_clr marks the first product of
    // a slot, the addend is forced to zero so the slot restarts; otherwise the
    // previous partial sum is fed back, giving per-slot accumulation.
    wire [31:0] cin = acc_clr ? 32'd0 : psum;     // +0.0 in every mode

    // Multiply-accumulate unit: acc_out = mul_a * mul_b + cascade_sum_in, 12
    // cycles after the operands are applied. It also delays mul_a / mul_b by one
    // cycle (cascade_mula_out / cascade_mulb_out) for the neighbouring PEs.
    fane_mac_simd mac_unit (
        .clk              (clk),                    // clock
        .rst_n            (rst_n),                  // active-low reset
        .ce               (ce),                     // pipeline clock enable
        .ce_a_1           (ce_a_1),                 // A operand register enable
        .ce_a_2           (1'b1),         // unused inside the MAC
        .ce_b_1           (ce_b_1),                 // B operand register enable
        .ce_b_2           (1'b1),         // unused inside the MAC
        .mode_control     (mode_control),           // number format of the packed operands
        .mul_a            (in_a),                   // activation operand
        .mul_b            (in_b),                   // weight operand
        .cascade_sum_in   (cin),                    // addend: 0 on a new accumulation, else the slot's psum
        .acc_out          (psum),                   // running sum of the current slot (fed back as cin)
        .cascade_mula_out (out_a),                  // activation forwarded to the right PE
        .cascade_mulb_out (out_b)                   // weight forwarded to the bottom PE
    );

    // forward the clear flag with the same timing as cascade_mula_out_reg
    reg clr_q;
    always @(posedge clk or negedge rst_n) begin
        // reset: no clear pending; otherwise capture acc_clr on each enabled cycle
        if (!rst_n)  clr_q <= 1'b0;
        else if (ce) clr_q <= acc_clr;
    end
    assign acc_clr_out = clr_q;

endmodule