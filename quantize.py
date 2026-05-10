import argparse
import onnx
from onnxconverter_common import float16

def convert_to_fp16(input_path, output_path):
    """Function to be called from other Python projects."""
    model = onnx.load(input_path)
    
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        # Common blocks that usually stay float32 for stability
        op_block_list=['RandomNormalLike', 'Range']
    )
    
    onnx.save(model_fp16, output_path)
    print(f"Successfully saved FP16 model to: {output_path}")

if __name__ == "__main__":
    # This part runs when you execute the script directly
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument("input", help="Path to input .onnx file")
    parser.add_argument("output", help="Path to save output .onnx file")
    
    args = parser.parse_args()
    convert_to_fp16(args.input, args.output)
