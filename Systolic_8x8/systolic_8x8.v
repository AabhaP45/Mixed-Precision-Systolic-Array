// =============================================================================
// systolic_8x8_fane_simd -- output-stationary systolic array (N x N PEs, default N = 8)
//
//   PE(r,c) accumulates  sum_k A[r][k] * B[k][c]
//
//   * A moves left -> right, B moves top -> bottom, 1 cycle per hop.
//   * Standard skew: A[r][k] enters row r at cycle  k + r ;
//                    B[k][c] enters column c at cycle k + c .
//                    PE(r,c) then sees both at cycle  k + r + c .
//   * psum_flat holds the psum of EVERY PE.  After the 12-cycle MAC latency
//     every PE produces a new psum EVERY cycle (psum_flat updates each clock).
//   * The MAC loop is 12 cycles, so each PE has 12 interleaved accumulator
//     slots (slot = cycle mod 12).  Present 12 independent output tiles
//     round-robin: operand pair at cycle t = (tile t mod 12, k = t / 12).
//     psum of PE(r,c) at cycle  t + r + c + 12  is the running sum of that
//     tile after element k.
//   * acc_clr_in[r] marks the first element (k = 0) of a tile on row r; it
//     travels along the row with A (same skew), so drive it together with
//     in_a_flat[r].
// =============================================================================
// Ports:
//   in_a_flat   : one 32-bit packed operand per row (left edge)
//   in_b_flat   : one 32-bit packed operand per column (top edge)
//   acc_clr_in  : per-row "first element of a tile" flag, skewed with A
//   psum_flat   : current partial sum of every PE
// Each 32-bit word holds 1x FP32, 2x FP16, 4x FP8 or 8x FP4 values,
// depending on mode_control.
module systolic_8x8_fane_simd #(
    parameter N = 8  // array dimension (N rows x N columns of PEs)
)(
    input  wire        clk,             // clock
    input  wire        rst_n,           // active-low asynchronous reset
    input  wire        ce,              // clock enable shared by all PEs
    input  wire        ce_a_1,          // A operand register enable (keep high while computing)
    input  wire        ce_b_1,          // B operand register enable (keep high while computing)
    input  wire [1:0]  mode_control,    // 00:FP32, 01:2xFP16, 10:4xFP8, 11:8xFP4

    input  wire [N*32-1:0]   in_a_flat,     // row r    : [r*32 +: 32]
    input  wire [N*32-1:0]   in_b_flat,     // column c : [c*32 +: 32]
    input  wire [N-1:0]      acc_clr_in,    // row r    : [r]

    output wire [N*N*32-1:0] psum_flat      // PE (r,c) : [(r*N+c)*32 +: 32]
);

    // Per-PE interconnect, indexed [row][column]:
    //   a_out : activation forwarded to the PE on the right
    //   b_out : weight forwarded to the PE below
    //   c_out : accumulator-clear flag forwarded to the PE on the right
    //   psum  : the PE's running partial sum
    wire [31:0] a_out [0:N-1][0:N-1];
    wire [31:0] b_out [0:N-1][0:N-1];
    wire        c_out [0:N-1][0:N-1];
    wire [31:0] psum  [0:N-1][0:N-1];

    // Build the N x N grid of PEs. Each PE takes A from its left neighbour (or
    // the row input at the left edge) and B from the PE above (or the column
    // input at the top edge); the forwarding registers inside each PE provide the
    // 1-cycle hop per PE.
    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : row_loop
            for (c = 0; c < N; c = c + 1) begin : col_loop

                // A-side inputs: activation and clear flag come from the row input in
                // column 0, otherwise from the PE to the left.
                wire [31:0] current_in_a;
                wire        current_clr;
                if (c == 0) begin : a_edge
                    assign current_in_a = in_a_flat[(r*32) +: 32];
                    assign current_clr  = acc_clr_in[r];
                end else begin : a_int
                    assign current_in_a = a_out[r][c-1];
                    assign current_clr  = c_out[r][c-1];
                end

                // B-side input: from the column input in row 0, otherwise from the PE above.
                wire [31:0] current_in_b;
                if (r == 0) begin : b_edge
                    assign current_in_b = in_b_flat[(c*32) +: 32];
                end else begin : b_int
                    assign current_in_b = b_out[r-1][c];
                end

                // Processing element: multiply-accumulate with output-stationary partial sum.
                pe_fane_simd pe_inst (
                    .clk          (clk),
                    .rst_n        (rst_n),
                    .ce           (ce),
                    .ce_a_1       (ce_a_1),
                    .ce_b_1       (ce_b_1),
                    .mode_control (mode_control),
                    .in_a         (current_in_a),
                    .in_b         (current_in_b),
                    .acc_clr      (current_clr),
                    .out_a        (a_out[r][c]),
                    .out_b        (b_out[r][c]),
                    .acc_clr_out  (c_out[r][c]),
                    .psum         (psum[r][c])
                );

                // Expose this PE's partial sum in the flattened output bus.
                assign psum_flat[((r*N + c)*32) +: 32] = psum[r][c];
            end
        end
    endgenerate

endmodule