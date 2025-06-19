struct info {
    int kernel_v;
    int kernel_h;
    int number_input_channels;
};

void convolution_layer(
    int output_channels,
    struct info information,
    int dim1, // vertical
    int dim2, // horizontal
    int8_t input_tensor[][dim2][information.number_input_channels],
    int output_v, int output_h,
    int8_t node_weights[][information.kernel_v][information.kernel_h][information.number_input_channels],
    int8_t node_biases[],
    int8_t weight_zp[],
    int8_t output_zp[],
    int8_t input_zp[],
    bool padding,
    int strides[], // [vertical, horizontal]
    int32_t composite_scales[][2],
    int8_t output_tensor[][output_h][output_channels]
) {
    if (padding == 0) {
        int input_horizontal_pixel = 0;
        int input_vertical_pixel = 0;
        int input_channel_select = 0;
        int kernel_output_channel = 0;
        int kernel_horizontal_pixel = 0;
        int kernel_vertical_pixel = 0;
        int kernel_input_channel = 0;

        int horizontal_sweeps = ((dim2 - information.kernel_h) / strides[1]) + 1;
        if (horizontal_sweeps < 0) horizontal_sweeps = 0;

        int vertical_sweeps = ((dim1 - information.kernel_v) / strides[0]) + 1;
        if (vertical_sweeps < 0) vertical_sweeps = 0;

        int temp_store[output_channels];

        for (kernel_output_channel = 0; kernel_output_channel < output_channels; kernel_output_channel++) {
            for (int vertical = 0; vertical < vertical_sweeps; vertical++) {
                for (int horizontal = 0; horizontal < horizontal_sweeps; horizontal++) {

                    int64_t accumulate = 0;

                    for (kernel_input_channel = 0; kernel_input_channel < information.number_input_channels; kernel_input_channel++) {
                        input_channel_select = kernel_input_channel;

                        for (kernel_vertical_pixel = 0; kernel_vertical_pixel < information.kernel_v; kernel_vertical_pixel++) {
                            for (kernel_horizontal_pixel = 0; kernel_horizontal_pixel < information.kernel_h; kernel_horizontal_pixel++) {
                                int input_y = vertical * strides[0] + kernel_vertical_pixel;
                                int input_x = horizontal * strides[1] + kernel_horizontal_pixel;

                                int32_t inp_val = (int32_t)input_tensor[input_y][input_x][input_channel_select] - input_zp[0];
                                int32_t weight_val = (int32_t)node_weights[kernel_output_channel][kernel_vertical_pixel][kernel_horizontal_pixel][kernel_input_channel] - weight_zp[kernel_output_channel];

                                accumulate += (int64_t)inp_val * (int64_t)weight_val;
                            }
                        }
                    }

                    accumulate += (int32_t)node_biases[kernel_output_channel];

                    int32_t M = composite_scales[kernel_output_channel][0];
                    int32_t n = composite_scales[kernel_output_channel][1];

                    int64_t big_accumulate = ((accumulate * M) + (1ll << (n - 1))) >> n;

                    big_accumulate += output_zp[0];

                    if (big_accumulate > 127) big_accumulate = 127;
                    else if (big_accumulate < -128) big_accumulate = -128;

                    int8_t quantized_accumulate = (int8_t)big_accumulate;

                    output_tensor[vertical][horizontal][kernel_output_channel] = quantized_accumulate;
                }
            }
        }
    }
}
