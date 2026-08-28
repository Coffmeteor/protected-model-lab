import torch
from safetensors.torch import load_file, save_file

from common.safetensors_stream import write_streaming


def test_streaming_preserves_untouched_bytes_and_transforms(tmp_path):
    source=tmp_path/"source.safetensors"
    output=tmp_path/"output.safetensors"
    save_file({"a":torch.arange(6,dtype=torch.float32).reshape(2,3),"b":torch.arange(4,dtype=torch.float16)},source)
    write_streaming(source,output,lambda key,value,dtype: value+2 if key=="a" else None)
    result=load_file(output)
    torch.testing.assert_close(result["a"],torch.arange(6,dtype=torch.float32).reshape(2,3)+2)
    torch.testing.assert_close(result["b"],torch.arange(4,dtype=torch.float16))
