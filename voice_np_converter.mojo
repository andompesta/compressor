from python import Python
from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from utils.vector import DynamicVector
from utils.static_tuple import StaticTuple

alias TensorF32 = Tensor[DType.float32]

fn build_path(base_path: String, prompt: String) -> String:
    return base_path + prompt

fn main() raises:
    let DEST_PATH = "./data/speaker_embeddings/"
    let BASE_PATH = "/Users/scavallari/.cache/huggingface/hub/models--ylacombe--bark-large/snapshots/3610a72f025e842bc9031f50895f90edde6387d3/speaker_embeddings/v2/"
    let PROMTS = StaticTuple[3](
        "en_speaker_6_coarse_prompt.npy",
        "en_speaker_6_fine_prompt.npy",
        "en_speaker_6_semantic_prompt.npy",
    )

    Python.add_to_path(".")
    let sys = Python.import_module("sys")
    for p in sys.path:
        print(p)
    let np = Python.import_module("numpy")

    for i in range(len(PROMTS)):
        var prompt: String = PROMTS[i]
        let src_path = build_path(BASE_PATH, prompt)
        print("load file: " + src_path)
        print('')
        let np_array = np.load(src_path).astype(np.float32)
        print(np_array)

        var shape = DynamicVector[Int]()

        for s in np_array.shape:
            shape.append(int(s))

        let spec = TensorSpec(DType.float32, shape)
        var mojo_tensor = TensorF32(spec)

        if len(shape) == 1:
            for i in range(shape[0]):
                mojo_tensor[Index(i)] = np_array[i].to_float64().cast[DType.float32]()
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    mojo_tensor[Index(i, j,)] = np_array[i][j].to_float64().cast[DType.float32]()
        else:
            raise Error()

        prompt = prompt.replace(".npy", ".bin")
        let dest_path = build_path(DEST_PATH, prompt)
        print("write file: " + dest_path)
        mojo_tensor.tofile(dest_path)


        


