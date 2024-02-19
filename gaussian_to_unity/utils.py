import numpy as np
import struct
import os
import torch
from sklearn.cluster import KMeans

kScaler = (1 << 21) - 1

VectorFormats = {'Float32':0, 'Norm16':1, 'Norm11':2, 'Norm6':3}

SHFormats = {'Float32':0, 'Float16':1, 'Norm11':2, 'Norm6':3, 
             'Cluster64k':4, 'Cluster32k':5, 'Cluster16k':6, 
             'Cluster8k':7, 'Cluster4k':8}

ColorFormats = {'Float32x4':0, 'Float16x4':1, 'Norm8x4':2, 'BC7':3}

sh_format_size = {'Float32': -1,
        'Float16': -1,
        'Norm11': -1,
        'Norm6': -1,
        'Cluster64k': 64 * 1024,
        'Cluster32k': 32 * 1024,
        'Cluster16k': 16 * 1024,
        'Cluster8k': 8 * 1024,
        'Cluster4k': 4 * 1024,
    }

passes_over_data = {
        'Cluster64k': 0.3,
        'Cluster32k': 0.4,
        'Cluster16k': 0.5,
        'Cluster8k': 0.8,
        'Cluster4k': 1.2,
    }

sh_format_to_numpy = {'Float16': np.float16,
                      'Float32': np.float32,
                      'Norm11': np.uint32,
                      'Norm6': np.uint16}

kTextureWidth = 2048
kShDim = 15 * 3
kBatchSize = 2048

def morton_part1by2(x):
    x &= 0x1fffff
    x = (x ^ (x << 32)) & 0x1f00000000ffff
    x = (x ^ (x << 16)) & 0x1f0000ff0000ff
    x = (x ^ (x << 8))  & 0x100f00f00f00f00f
    x = (x ^ (x << 4))  & 0x10c30c30c30c30c3
    x = (x ^ (x << 2))  & 0x1249249249249249
    return x

def morton_encode3(v):
    return (morton_part1by2(v[2]) << 2) | (morton_part1by2(v[1]) << 1) | morton_part1by2(v[0])

def reorder_morton_job(bounds_min, inv_bounds_size, splat_data):
    n = len(splat_data)
    order = np.zeros((n, 2), dtype=np.uint64)
    splat_data  = splat_data.cpu().numpy()
    pos = (splat_data - bounds_min) * inv_bounds_size * kScaler
    ipos = pos.astype(np.uint32)

    for index in range(n):
        code = morton_encode3(ipos[index])
        order[index] = (code, index)
    
    return np.array(sorted(order, key=lambda element: element[0]))

    #return order

def encode_float3_to_norm16(v):
    x = (v[:, 0] * 65535.5).astype(np.uint64)
    y = (v[:, 1] * 65535.5).astype(np.uint64)
    z = (v[:, 2] * 65535.5).astype(np.uint64)
    return x | (y << 16) | (z << 32)

def encode_float3_to_norm11(v):
    x = (v[:, 0] * 2047.5).astype(np.uint32)
    y = (v[:, 1] * 1023.5).astype(np.uint32)
    z = (v[:, 2] * 2047.5).astype(np.uint32)
    return (x | (y << 11) | (z << 21))

def encode_float3_to_norm655(v):
    x = (v[:,0] * 63.5).astype(np.uint32)
    y = (v[:,1] * 31.5).astype(np.uint32)
    z = (v[:,2] * 31.5).astype(np.uint32)
    return (x | (y << 6) | (z << 11)).astype(np.uint16)

def encode_float3_to_norm565(v):
    x = (v[:,0] * 31.5).astype(np.uint32)
    y = (v[:,1] * 63.5).astype(np.uint32)
    z = (v[:,2] * 31.5).astype(np.uint32)
    return (x | (y << 5) | (z << 11)).astype(np.uint16)

def encode_quat_to_norm10(v):
    x = (v[:,0] * 1023.5).astype(np.uint64)
    y = (v[:,1] * 1023.5).astype(np.uint64)
    z = (v[:,2] * 1023.5).astype(np.uint64)
    w = (v[:,3] * 3.5).astype(np.uint64)
    return (x | (y << 10) | (z << 20) | (w << 30)).astype(np.uint32)

def encode_vector(v, vector_format):
    v = np.clip(v, 0, 1)
    
    if vector_format == "Float32":
        v = v.astype(np.float32)
        return v
        
    elif vector_format == "Norm16":
        enc = encode_float3_to_norm16(v)
        low_bits = (enc & 0xFFFFFFFF).astype(np.uint32)
        high_bits = (enc >> 32).astype(np.uint16)
        return np.core.records.fromarrays([low_bits, high_bits], 
                                  names='low_bits,high_bits', 
                                  formats = 'I,H')
    elif vector_format == "Norm11":
        enc = encode_float3_to_norm11(v)
        return enc

    elif vector_format == "Norm6":
        enc = encode_float3_to_norm655(v)
        return enc
    
    elif vector_format == "Norm565":
        enc = encode_float3_to_norm565(v)
        return enc
    
    elif vector_format == "Norm10":
        enc = encode_quat_to_norm10(v)
        return enc
    
    else:
        raise ValueError(f"Unknown vector format: {vector_format}")


def normalize_sh_chunks(chunks):

    bounds_max = np.max(chunks, axis=(1,2))
    bounds_min = np.min(chunks, axis=(1,2))

    bounds_max_expanded = bounds_max[:, np.newaxis, np.newaxis, :]
    bounds_min_expanded = bounds_min[:, np.newaxis, np.newaxis, :]

    # Normalize
    normalized = (chunks - bounds_min_expanded) / (bounds_max_expanded - bounds_min_expanded)

    # Compute chunk bounds
    chunk_bounds = np.column_stack((bounds_min_expanded, bounds_max_expanded))

    return normalized, chunk_bounds

def normalize_sh(data_in):

    bounds_max = np.max(data_in, axis=(0,1))
    bounds_min = np.min(data_in, axis=(0,1))

    bounds_max_expanded = bounds_max[np.newaxis, np.newaxis, :]
    bounds_min_expanded = bounds_min[np.newaxis, np.newaxis, :]

    # Normalize
    normalized = (data_in - bounds_min_expanded) / (bounds_max_expanded - bounds_min_expanded)

    # Compute chunk bounds
    chunk_bounds = np.concatenate((bounds_min_expanded, bounds_max_expanded), axis=0)

    return normalized, chunk_bounds



def normalize_chunks(chunks):
    bounds_min = np.min(chunks, axis=1)
    bounds_max = np.max(chunks, axis=1)

    bounds_min_expanded = bounds_min[:, np.newaxis, :]
    bounds_max_expanded = bounds_max[:, np.newaxis, :]

    normalized = (chunks - bounds_min_expanded) / (bounds_max_expanded - bounds_min_expanded)

    chunk_bounds = np.column_stack((bounds_min_expanded, bounds_max_expanded))

    return normalized, chunk_bounds

def normalize(data):
    min, max = calculate_bounds(data)
    normalized = (data - min) / (max - min)
    bounds = np.stack((min, max))

    return normalized, bounds

def create_chunks(data_in, gaussianCount, chunk_size, sh=False):
    
    n_chunks = gaussianCount // chunk_size
    
    if (not sh):
        chunks_array = data_in[:n_chunks * chunk_size].reshape(-1, chunk_size, data_in.shape[1])
        normalized, chunk_bounds = normalize_chunks(chunks_array)
        normalized = normalized.reshape(-1, data_in.shape[1])

        # Include the rest of the data that does not fit in a chunk
        remaining = data_in[n_chunks * chunk_size:].reshape(-1, data_in.shape[1])
        remaining_norm, remaining_bounds = normalize(remaining)

        normalized = np.concatenate((normalized, remaining_norm), axis=0)
        chunk_bounds = np.concatenate((chunk_bounds, remaining_bounds[np.newaxis,:,:]), axis=0)

    else:
        chunks_array = data_in[:n_chunks * chunk_size].reshape(-1, chunk_size, data_in.shape[1], data_in.shape[2])
        normalized, chunk_bounds = normalize_sh_chunks(chunks_array)
        normalized = normalized.reshape(-1, data_in.shape[1], data_in.shape[2])
    
        # Include the rest of the data that does not fit in a chunk
        remaining = data_in[n_chunks * chunk_size:].reshape(-1, data_in.shape[1], data_in.shape[2])
        remaining_norm, remaining_bounds = normalize_sh(remaining)

        normalized = np.concatenate((normalized, remaining_norm), axis=0)
        chunk_bounds = np.concatenate((chunk_bounds, remaining_bounds[np.newaxis, :,:,:]), axis=0)

    return normalized, chunk_bounds

def calculate_bounds(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return mins, maxs

# Un poco distinta a Unity
def create_positions_asset(means3D_sorted, basepath, format='Norm11', idx=-1, one_file=False):
    if one_file:
        if idx == 0:
            if os.path.exists(os.path.join(basepath, f"position_data.bytes")):
                os.remove(os.path.join(basepath, f"position_data.bytes"))

        path = os.path.join(basepath, f"position_data.bytes")

        with open(path, 'ab') as f:
            for mean3d in means3D_sorted:
                f.write(encode_vector(means3D_sorted, format).tobytes())
    else:
        output_folder = os.path.join(os.path.dirname(basepath), "positions")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{idx}.bytes")

        with open(path, 'wb') as f:
            f.write(encode_vector(means3D_sorted, format).tobytes())


def f32tof16(f32):
    f16 = np.float16(f32)
    return f16.view(np.uint16)

def create_others_asset(rotations, scales, basepath, scale_format, idx=-1):

    if idx == -1:
        path = os.path.join(basepath, f"others.bytes")
        if os.path.exists(path):
            os.remove(path)
    else:
        output_folder = os.path.join(os.path.dirname(basepath), "others")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{idx}.bytes")

    with open(path, 'wb') as f:

        encoded_rotations = encode_vector(rotations, "Norm10")
        encoded_scales = encode_vector(scales, scale_format)
        
        interleaved = np.empty((encoded_rotations.size + encoded_scales.size,), dtype=encoded_rotations.dtype)
        
        # Interleave the data
        interleaved[::2] = encoded_rotations
        interleaved[1::2] = encoded_scales

        f.write(interleaved.tobytes())

def f32tof16(f32):
    f16 = np.float16(f32)
    return f16.view(np.uint16).astype(np.uint32)

def create_chunks_asset(pos_chunks, scale_chunks, basepath, idx=0, one_file=False):

    if one_file:
        path = os.path.join(basepath, f"chunk_data.bytes")
        mode = 'ab' if idx > 0 else 'wb'
    else:
        output_folder = os.path.join(os.path.dirname(basepath), "chunks")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{idx}.bytes")
        mode = 'wb'
    
    sclX = (f32tof16(scale_chunks[:, 0, 0]) | (f32tof16(scale_chunks[:, 1, 0]) << 16)).view(np.float32)
    sclY = (f32tof16(scale_chunks[:, 0, 1]) | (f32tof16(scale_chunks[:, 1, 1]) << 16)).view(np.float32)
    sclZ = (f32tof16(scale_chunks[:, 0, 2]) | (f32tof16(scale_chunks[:, 1, 2]) << 16)).view(np.float32)
    
    # Packing data
    packed_data = np.column_stack((pos_chunks[:, :, 0], pos_chunks[:, :, 1], pos_chunks[:, :, 2], sclX, sclY, sclZ))

    # Write to file
    with open(path, mode) as f:
        f.write(packed_data.tobytes())


def create_chunks_static_asset(col_chunks, shs_chunks, basepath):

    path = os.path.join(basepath, "chunks_static.bytes")
    if os.path.exists(path):
        os.remove(path)
    mode = 'ab'
    
    colR = f32tof16(col_chunks[:, 0, 0]) | (f32tof16(col_chunks[:, 1, 0]) << 16)
    colG = f32tof16(col_chunks[:, 0, 1]) | (f32tof16(col_chunks[:, 1, 1]) << 16)
    colB = f32tof16(col_chunks[:, 0, 2]) | (f32tof16(col_chunks[:, 1, 2]) << 16)
    colA = f32tof16(col_chunks[:, 0, 3]) | (f32tof16(col_chunks[:, 1, 3]) << 16)

    shR = f32tof16(shs_chunks[:, 0, 0]) | (f32tof16(shs_chunks[:, 1, 0]) << 16)
    shG = f32tof16(shs_chunks[:, 0, 1]) | (f32tof16(shs_chunks[:, 1, 1]) << 16)
    shB = f32tof16(shs_chunks[:, 0, 2]) | (f32tof16(shs_chunks[:, 1, 2]) << 16)

    packed_data = np.dstack((colR, colG, colB, colA, shR, shG, shB))
    
    with open(path, mode) as f:
        f.write(packed_data.tobytes())

def normalize_swizzle_rotation(wxyz):
    normalized = np.divide(wxyz,np.linalg.norm(wxyz, axis=1).reshape(-1, 1))
    normalized = np.roll(normalized, -1)
    return normalized

def pack_smallest_3_rotation(q):
    abs_q = np.abs(q)
    index = np.argmax(abs_q, axis=1)
    n = q.shape[1]
    rolled_indices = np.zeros((q.shape[0], n), dtype=np.int32)

    rolled_indices[index == 0, :] = [1, 2, 3, 0]
    rolled_indices[index == 1, :] = [0, 2, 3, 1]
    rolled_indices[index == 2, :] = [0, 1, 3, 2]
    rolled_indices[index == 3, :] = [0, 1, 2, 3]

    q_rolled = q[np.arange(q.shape[0])[:, np.newaxis], rolled_indices]
    signs = np.sign(q_rolled[:, 3])
    three = q_rolled[:, :3] * signs[:, np.newaxis]
    three = (three * np.sqrt(2)) * 0.5 + 0.5
    index = index / 3.0

    return np.column_stack((three, index))


def linear_scale(log_scale):
    return np.abs(np.exp(log_scale))

def sh0_to_color(dc0):
    kSH_C0 = 0.2820948
    return dc0 * kSH_C0 + 0.5

def linealize(rot, scale):
    # Rotation processing
    rot = normalize_swizzle_rotation(rot)
    rot = pack_smallest_3_rotation(rot)

    # Scale processing
    scale = linear_scale(scale)
    scale = scale ** (1.0 / 8.0)

    return rot, scale

# Define a custom sorting key
def sort_key(filename):
    # Remove the '.bytes' extension and convert the remaining string to an integer
    return int(filename.replace('.bytes', ''))

def create_one_file(basepath, splat_count=0, chunk_count=0, frame_time=1/20, args=None):

    # Current format
    # 1- Metadata
    # 2- Static data
    # 3- Dynamic data, intercalated positions and chunks

    positions_path = os.path.join(basepath, "positions")
    others_path = os.path.join(basepath, "others")
    chunks_path = os.path.join(basepath, "chunks")
    
    data = []
    
    # ---- Header information -----

    format_version = 20231006

    color_width, color_height = calc_texture_size(splat_count)

    frame_count = len(os.listdir(positions_path))

    data.append(struct.pack('I', format_version)) # Format version
    data.append(struct.pack('I', splat_count)) # Splat count
    data.append(struct.pack('f', frame_time)) # Frame time
    data.append(struct.pack('I', frame_count)) # Frame count
    data.append(struct.pack('I', chunk_count)) # Chunk count
    data.append(struct.pack('I', VectorFormats[args.pos_format])) # Position format
    data.append(struct.pack('I', VectorFormats[args.scale_format])) # Scale format
    data.append(struct.pack('I', SHFormats[args.sh_format])) # SH format
    data.append(struct.pack('I', ColorFormats[args.col_format])) # Color format 
    data.append(struct.pack('I', color_width)) # Color width
    data.append(struct.pack('I', color_height)) # Color height
    data.append(struct.pack('I', int(args.include_others))) # Include dynamic rotations and scaling
    
    # Transforms
    data.append(struct.pack('f', args.pos_offset[0])) # Position offset x
    data.append(struct.pack('f', args.pos_offset[1])) # Position offset y
    data.append(struct.pack('f', args.pos_offset[2])) # Position offset z
    data.append(struct.pack('f', args.rot_offset[0])) # Rotation offset x
    data.append(struct.pack('f', args.rot_offset[1])) # Rotation offset y
    data.append(struct.pack('f', args.rot_offset[2])) # Rotation offset z
    data.append(struct.pack('f', args.scale[0])) # Scale x
    data.append(struct.pack('f', args.scale[1])) # Scale y
    data.append(struct.pack('f', args.scale[2])) # Scale z

    static_info = ["chunks_static.bytes", "colors.bytes", "shs.bytes"]

    if (not args.include_others):
        static_info.append("others.bytes")
    
    # ---- Static data ----
    
    for info in static_info:
        with open(os.path.join(basepath, info), 'rb') as f:
            size = os.path.getsize(os.path.join(basepath, info))
            data.append(struct.pack('I', size))
            data.append(f.read())
    
    # ---- Dynamic data ----
    
    # Read all the files intercalated
    pos_size = os.path.getsize(os.path.join(positions_path, os.listdir(positions_path)[0]))
    data.append(struct.pack('I', pos_size))

    chunk_size = os.path.getsize(os.path.join(chunks_path, os.listdir(chunks_path)[0]))
    data.append(struct.pack('I', chunk_size))
    
    if (args.include_others):
        others_size = os.path.getsize(os.path.join(others_path, os.listdir(others_path)[0]))
        data.append(struct.pack('I', others_size))
    
    for position_file in sorted(os.listdir(positions_path), key=sort_key):
        with open(os.path.join(positions_path, position_file), 'rb') as f:
            data.append(f.read())
    
        with open(os.path.join(chunks_path, position_file), 'rb') as f:
            data.append(f.read())
        
        if (args.include_others):
            with open(os.path.join(others_path, position_file), 'rb') as f:
                data.append(f.read())
    
    # Write the data to a single file
    file_name = os.path.join(os.path.dirname(basepath), f"{args.save_name}.bytes")

    with open(file_name, 'wb') as f:
        for chunk in data:
            f.write(chunk)
    print("File created: ", file_name)


def create_one_file_chunk_pos(basepath):

    positions_path = os.path.join(os.path.dirname(basepath), "positions")
    chunks_path = os.path.join(os.path.dirname(basepath), "chunks")

    pos_data = []
    chunk_data = []

    for position_file in sorted(os.listdir(positions_path)):
        with open(os.path.join(positions_path, position_file), 'rb') as f1:
            pos_data.append(f1.read())

        with open(os.path.join(chunks_path, position_file), 'rb') as f2:
            chunk_data.append(f2.read())
    
    pos_file_name = os.path.join(os.path.dirname(basepath), "position_data.bytes")
    chunk_file_name = os.path.join(os.path.dirname(basepath), "chunk_data.bytes")

    with open(pos_file_name, 'wb') as f1:
        for chunk in pos_data:
            f1.write(chunk)
    
    with open(chunk_file_name, 'wb') as f2:
        for chunk in chunk_data:
            f2.write(chunk)

# Color utils
def decode_morton2D_16x16(t):
    t = (t & 0xFF) | ((t & 0xFE) << 7)
    t &= 0x5555
    t = (t ^ (t >> 1)) & 0x3333
    t = (t ^ (t >> 2)) & 0x0f0f
    return t & 0xF, t >> 8

def splat_index_to_texture_index(idx):
    xy = decode_morton2D_16x16(idx)
    width = kTextureWidth // 16
    idx >>= 8
    x = (idx % width) * 16 + xy[0]
    y = (idx // width) * 16 + xy[1]
    return y * kTextureWidth + x

def calc_texture_size(splat_count):
    height =  (splat_count + kTextureWidth -1) // kTextureWidth
    blockHeight = 16
    height = (height + blockHeight - 1) // blockHeight * blockHeight

    return kTextureWidth, height


def encode_color(v, color_format):
    v = np.clip(v, 0, 1)

    if color_format == "Float32x4":
        return v.astype(np.float32)

    elif color_format == "Float16x4":
        return v.astype(np.float16)

    elif color_format == "Norm8x4":
        v = (v * 255.5).astype(np.uint32)
        enc = v[:, 0] | (v[:, 1] << 8) | (v[:, 2] << 16) | (v[:, 3] << 24)
        return enc


def create_colors_asset(splat_count, col, basepath="/", color_format="Norm8x4"):

    width, height = calc_texture_size(splat_count)

    data = np.zeros((height * width, 4), dtype=np.float32)

    idx = np.linspace(0, splat_count - 1, splat_count).astype(np.uint32)
    morton_idx = splat_index_to_texture_index(idx.copy())

    col_ref = col.copy()
    # reorder and to numpy
    data[morton_idx] = col_ref[idx]
    
    file_name = os.path.join(basepath, "colors.bytes")
    
    if os.path.exists(file_name):
        os.remove(file_name)

    encoded_data = encode_color(data, color_format).tobytes()

    with open(file_name, 'wb') as f:
        f.write(encoded_data)


def cluster_shs(shs, sh_format="Norm6"):

    sh_count = sh_format_size[sh_format]

    if sh_count == -1:
        return shs, None

    kmeans = KMeans(n_clusters=sh_count)
    sh_indices = kmeans.fit_predict(shs)
    sh_means = kmeans.cluster_centers_

    return sh_means, sh_indices

def create_sh_asset(splat_count, shs, sh_data_format="Norm6", basepath="/"):
    
    sh_count = sh_format_size[sh_data_format]

    # For shs Norm6 is done as Norm565
    if sh_data_format == 'Norm6':
        sh_data_format = 'Norm565'

    if sh_count == -1:
        sh_count = splat_count

    sh_len = shs.shape[1]

    path = os.path.join(basepath, f"shs.bytes")

    if os.path.exists(path):
        os.remove(path)

    if sh_data_format != "Norm11":
        padding = np.zeros((shs.shape[0], 1 , 3), dtype=shs.dtype)
        shs = np.concatenate((shs, padding), axis=1)
    
    shs = shs.reshape(-1, 3)
    
    encoded_shs = encode_vector(shs, sh_data_format)

    with open(path, 'wb') as f:
        f.write(encoded_shs.tobytes())
    
    
def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def square_centered01(x):
    x -= 0.5
    x *= x * np.sign(x)
    return x * 2.0 + 0.5

def linealize_colors(dc, opacity):
    dc = sh0_to_color(dc)
    opacity = square_centered01(sigmoid(opacity))
    return dc, opacity

def create_deleted_mask(deleted_path):
    with open(deleted_path, 'rb') as f:
        deleted = np.fromfile(f, dtype=np.uint8)
    mask = np.unpackbits(deleted).astype(bool)
    # to torch
    #mask = torch.from_numpy(mask)
    return mask


if __name__=="__main__":
    deleted_path = "output/scene_deleted.bytes"
    mask = create_deleted_mask(deleted_path)
    print("Testing")
    