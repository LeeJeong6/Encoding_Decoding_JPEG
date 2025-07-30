import heapq
from collections import defaultdict, Counter
import bitstring
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import argparse
def pad_to_multiple_of_8(arr):
    """입력 배열(arr)을 8의 배수 크기로 패딩"""
    h, w = arr.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return arr  # 이미 8의 배수이면 그대로 반환
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    print(f"[패딩 적용] 원본 크기 ({h}x{w}) → 패딩 후 ({padded.shape[0]}x{padded.shape[1]})")
    return padded
def memory_kb(arrays):
    """넘파이 배열 리스트의 총 메모리 (KB)"""
    total = 0
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            total += arr.nbytes
    return total / 1024

def rle_memory_kb(rle_blocks):
    """RLE 리스트 용량 추정 (KB).
       각 (zero_run, coeff) 쌍을 int16 * 2 로 가정"""
    total_pairs = sum(len(block) for block in rle_blocks)
    return total_pairs * 2 * 2 / 1024   # 2개 값 * 2바이트(int16)

def bitstream_memory_kb(bitstream_list):
    """허프만 비트스트림 용량 (KB)"""
    bits = sum(len(bs) for bs in bitstream_list)
    return bits / 8 / 1024

QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
QC = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])
QY_scaled = np.array([
    [24, 16, 15, 24, 36, 60, 76, 91],
    [18, 18, 21, 28, 39, 87, 90, 82],
    [21, 19, 24, 36, 60, 85, 103, 84],
    [21, 25, 33, 43, 76, 130, 120, 93],
    [27, 33, 55, 84, 102, 163, 154, 115],
    [36, 52, 82, 96, 121, 156, 169, 138],
    [73, 96, 117, 130, 154, 181, 180, 151],
    [108, 138, 142, 147, 168, 150, 154, 148]
])

QC_scaled = np.array([
    [25, 27, 36, 70, 148, 148, 148, 148],
    [27, 31, 39, 99, 148, 148, 148, 148],
    [36, 39, 84, 148, 148, 148, 148, 148],
    [70, 99, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148]
])
def DCT(block):
    return dct(dct(block, type=2, norm="ortho").T, type=2, norm="ortho")

def IDCT(block):    
    return idct(idct(block, type=2, norm="ortho").T, type=2, norm="ortho")

def process_img(image, Q):
    """각 채널을 DCT 후 양자화"""
    h, w = image.shape 
    channel_dctq = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8): 
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8] - 128
            dct_block = DCT(block)
            q_block = np.round(dct_block / Q)
            channel_dctq[i:i+8, j:j+8] = q_block
    return channel_dctq

def zigzag_indices(n=8):
    """8x8 인덱스 설정"""
    indices = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            for i in range(s+1):
                j = s - i
                if i < n and j < n:
                    indices.append((i, j))
        else:
            for i in range(s, -1, -1):
                j = s - i
                if i < n and j < n:
                    indices.append((i, j))
    return indices

def zigzag_scan(block):
    """8x8패치를 한줄로 펼치기"""
    return [block[i, j] for i, j in zigzag_indices()]

def RLE(coeffs):
    """Run Length Encoding"""
    result = []
    zeros = 0
    for c in coeffs[1:]:
        if c == 0:
            zeros += 1
        else:
            result.append((zeros, int(c)))
            zeros = 0
    result.append((0, 0))
    return [coeffs[0]] + result

def jpeg_encode(channel):
    """각 채널 RLE"""
    h, w = channel.shape
    encoded_blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            coeffs = zigzag_scan(block)
            rle = RLE(coeffs)
            encoded_blocks.append(rle)
    return encoded_blocks 

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_table):
    """빈도 테이블을 기반으로 Huffman 트리 구축"""
    heap = []
    for symbol, freq in freq_table.items():
        heapq.heappush(heap, HuffmanNode(symbol, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    return heap[0]

def generate_huffman_codes(node, current_code="", codes=None):
    """Huffman 트리에서 코드 생성"""
    if codes is None:
        codes = {}
    
    if node.symbol is not None:
        codes[node.symbol] = current_code or "0"
    
    if node.left:
        generate_huffman_codes(node.left, current_code + "0", codes)
    if node.right:
        generate_huffman_codes(node.right, current_code + "1", codes)
    
    return codes

def huffman_encode(rle_data):
    """RLE 데이터를 Huffman 코딩으로 압축"""
    dc_coeffs = [block[0] for block in rle_data]
    ac_coeffs = [tuple(item) for block in rle_data for item in block[1:]]
    
    dc_freq = Counter(dc_coeffs)
    ac_freq = Counter(ac_coeffs)
    
    dc_tree = build_huffman_tree(dc_freq)
    ac_tree = build_huffman_tree(ac_freq)
    dc_codes = generate_huffman_codes(dc_tree)
    ac_codes = generate_huffman_codes(ac_tree)
    
    bitstream = bitstring.BitArray()
    for block in rle_data:
        bitstream.append(f"0b{dc_codes[block[0]]}")
        for ac in block[1:]:
            bitstream.append(f"0b{ac_codes[tuple(ac)]}")
    
    return bitstream, dc_codes, ac_codes





#########################################################################
"""여기부터 Decoding"""



def huffman_decode(bitstream, dc_codes, ac_codes, block_count):
    """Huffman 비트스트림을 디코딩하여 RLE 데이터 복원"""
    inv_dc_codes = {v: k for k, v in dc_codes.items()}
    inv_ac_codes = {v: k for k, v in ac_codes.items()}
    decoded_blocks = []
    bits = bitstring.BitStream(bitstream)
    
    for _ in range(block_count):
        block = []
        code = ""
        while bits.pos < bits.len:
            code += bits.read("bin:1")
            if code in inv_dc_codes:
                block.append(inv_dc_codes[code])
                break
        while bits.pos < bits.len:
            code = ""
            while bits.pos < bits.len:
                code += bits.read("bin:1")
                if code in inv_ac_codes:
                    ac = inv_ac_codes[code]
                    block.append(ac)
                    if ac == (0, 0):
                        break
                    break
            if ac == (0, 0):
                break
        decoded_blocks.append(block)
    
    return decoded_blocks

def compare_rle(original_rle, decoded_rle):
    """원본 RLE와 디코딩된 RLE 비교"""
    if len(original_rle) != len(decoded_rle):
        print(f"Length mismatch: Original {len(original_rle)}, Decoded {len(decoded_rle)}")
        return False
    
    for i, (orig_block, dec_block) in enumerate(zip(original_rle, decoded_rle)):
        if orig_block != dec_block:
            print(f"Mismatch in block {i}: Original {orig_block}, Decoded {dec_block}")
            return False
    
    print("RLE data matches perfectly!")
    return True

def inverse_rle(rle_block):
    """RLE 데이터를 8x8 블록의 계수로 복원"""
    coeffs = [rle_block[0]]
    for zeros, value in rle_block[1:]:
        coeffs.extend([0] * zeros)
        if (zeros, value) == (0, 0):
            coeffs.extend([0] * (64 - len(coeffs)))
            break
        coeffs.append(value)
    return coeffs[:64]

def inverse_zigzag(coeffs, zigzag_indices):
    """지그재그 스캔된 계수를 8x8 블록으로 복원"""
    block = np.zeros((8, 8), dtype=np.float32)
    for idx, (i, j) in enumerate(zigzag_indices):
        block[i, j] = coeffs[idx]
    return block

def inverse_quantize(block, quant_table):
    """양자화 역처리"""
    return block * quant_table

def reconstruct_image(blocks, height, width):
    """블록을 이미지로 재구성"""
    img = np.zeros((height, width), dtype=np.float32)
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            img[i:i+8, j:j+8] = blocks[block_idx]
            block_idx += 1
    return img

def ycbcr_to_rgb(y, cb, cr):
    """YCbCr을 RGB로 변환"""
    # Cb, Cr을 512x512로 업샘플링
    cb = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
    cr = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)
    
    # RGB 변환 공식
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    
    # 0~255 범위로 클리핑
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def main(img_path):

    img = Image.open(img_path)
    width, height = img.size
    

    #YCbCr 변환
    ycbcr_img = img.convert("YCbCr")
    Y, Cb, Cr = ycbcr_img.split()
    
    Y = np.array(Y, dtype=np.float32)
    Cb = np.array(Cb, dtype=np.float32)
    Cr = np.array(Cr, dtype=np.float32)

    # 8의 배수가 아니면 패딩
    Y = pad_to_multiple_of_8(Y)
    Cb = pad_to_multiple_of_8(Cb)
    Cr = pad_to_multiple_of_8(Cr)

    #Downsampliing
    Cb_ds = Cb[::2, ::2]    
    Cr_ds = Cr[::2, ::2]

    # 인코딩 - DCT변환
    Y_dctq = process_img(Y, QY_scaled)
    Cb_dctq = process_img(Cb_ds, QC_scaled)
    Cr_dctq = process_img(Cr_ds, QC_scaled)

   
    # 지그재그 스캔 + RLE인코딩
    Y_encoded = jpeg_encode(Y_dctq)
    Cb_encoded = jpeg_encode(Cb_dctq)
    Cr_encoded = jpeg_encode(Cr_dctq)
    
    # Huffman 인코딩
    Y_bitstream, Y_dc_codes, Y_ac_codes = huffman_encode(Y_encoded)
    Cb_bitstream, Cb_dc_codes, Cb_ac_codes = huffman_encode(Cb_encoded)
    Cr_bitstream, Cr_dc_codes, Cr_ac_codes = huffman_encode(Cr_encoded)

    # 메모리 차지 용량 비교
    orig_size = os.path.getsize(img_path) / 1024
    ycbcr_size = memory_kb([Y, Cb, Cr])
    downsampled_size = memory_kb([Y, Cb_ds, Cr_ds])
    quantized_size = memory_kb([Y_dctq, Cb_dctq, Cr_dctq])
    rle_size = rle_memory_kb(Y_encoded + Cb_encoded + Cr_encoded)
    huffman_size = bitstream_memory_kb([Y_bitstream, Cb_bitstream, Cr_bitstream])

    sizes = {
        "원본 JPEG 파일": orig_size,
        "YCbCr 분리": ycbcr_size,
        "다운샘플링": downsampled_size,
        "양자화 후": quantized_size,
        "RLE 후": rle_size,
        "허프만 후": huffman_size
    }

    print("단계별 JPEG 압축률:")
    for stage, size in sizes.items():   
        print(f"{stage}: {size:.2f} KB")


    """여기까지가 encoding"""

    # Huffman 디코딩
    Y_decoded = huffman_decode(Y_bitstream, Y_dc_codes, Y_ac_codes, len(Y_encoded))
    Cb_decoded = huffman_decode(Cb_bitstream, Cb_dc_codes, Cb_ac_codes, len(Cb_encoded))
    Cr_decoded = huffman_decode(Cr_bitstream, Cr_dc_codes, Cr_ac_codes, len(Cr_encoded))
    
    # RLE 디코딩
    Y_coeffs = [inverse_rle(block) for block in Y_decoded]
    Cb_coeffs = [inverse_rle(block) for block in Cb_decoded]
    Cr_coeffs = [inverse_rle(block) for block in Cr_decoded]
    
    # 지그재그 역변환
    Y_blocks = [inverse_zigzag(coeffs, zigzag_indices()) for coeffs in Y_coeffs]
    Cb_blocks = [inverse_zigzag(coeffs, zigzag_indices()) for coeffs in Cb_coeffs]
    Cr_blocks = [inverse_zigzag(coeffs, zigzag_indices()) for coeffs in Cr_coeffs]
    
    # 양자화 역처리
    Y_dequant = [inverse_quantize(block, QY_scaled) for block in Y_blocks]
    Cb_dequant = [inverse_quantize(block, QC_scaled) for block in Cb_blocks]
    Cr_dequant = [inverse_quantize(block, QC_scaled) for block in Cr_blocks]
    
    # IDCT
    Y_pixels = [IDCT(block) + 128 for block in Y_dequant]
    Cb_pixels = [IDCT(block) + 128 for block in Cb_dequant]
    Cr_pixels = [IDCT(block) + 128 for block in Cr_dequant]
    
    # 이미지 재구성
    Y_recon = reconstruct_image(Y_pixels, width, width)
    Cb_recon = reconstruct_image(Cb_pixels, width//2, width//2)
    Cr_recon = reconstruct_image(Cr_pixels, width//2, width//2)
    
    # YCbCr → RGB 변환
    rgb_recon = ycbcr_to_rgb(Y_recon, Cb_recon, Cr_recon)
    
    # 원본 이미지와 비교 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(rgb_recon)
    ax2.set_title("Reconstructed Image")
    ax2.axis("off")
    
    plt.savefig("./compare JPEG.png")
    Image.fromarray(rgb_recon).save("./reconstructed.jpeg")


    # PSNR 계산 (품질 비교)
    mse = np.mean((np.array(img, dtype=np.float32) - rgb_recon) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG Compression Pipeline")
    parser.add_argument("--image_path", type=str, help="입력 이미지 경로")
    args = parser.parse_args()
    main(args.image_path)