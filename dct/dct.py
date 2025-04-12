import itertools
import numpy as np
import cv2
import traceback
import struct

# standard JPEG quantization matrix - luminance
quant = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

class DCT:
    def __init__(self):
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0

    # Encode image with text
    def encode_image(self, img, secret_msg):
        try:
            # Mã hóa chuỗi tiếng Việt sang bytes UTF-8
            secret_bytes = secret_msg.encode('utf-8')
            # Gắn tiền tố độ dài theo byte + dấu *
            length_prefix = str(len(secret_bytes)).encode('utf-8') + b'*'
            full_bytes = length_prefix + secret_bytes

            self.bitMess = self.toBits(full_bytes)

            row, col = img.shape[:2]
            self.oriRow, self.oriCol = row, col
            if (col // 8) * (row // 8) < len(self.bitMess):
                print("Error: Message too large to encode in image")
                return None

            if row % 8 != 0 or col % 8 != 0:
                img = self.addPadd(img, row, col)
            row, col = img.shape[:2]
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            bImg, gImg, rImg = cv2.split(img)
            bImg = np.float32(bImg)

            imgBlocks = [bImg[j:j+8, i:i+8] - 128
                         for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
            dctBlocks = [cv2.dct(block) for block in imgBlocks]
            quantizedDCT = [np.round(block / quant) for block in dctBlocks]

            messIndex = 0
            letterIndex = 0
            for quantizedBlock in quantizedDCT:
                if messIndex >= len(self.bitMess):
                    break
                DC = int(quantizedBlock[0][0])
                DC_bits = np.unpackbits(np.array([DC], dtype=np.int8).view(np.uint8))
                DC_bits[7] = int(self.bitMess[messIndex][letterIndex])
                quantizedBlock[0][0] = float(np.array([np.packbits(DC_bits)[0]], dtype=np.uint8).view(np.int8)[0])
                letterIndex += 1
                if letterIndex == 8:
                    letterIndex = 0
                    messIndex += 1

            sImgBlocks = [cv2.idct(block * quant) + 128 for block in quantizedDCT]
            sImg = []
            for chunkRowBlocks in self.chunks(sImgBlocks, col // 8):
                for rowBlockNum in range(8):
                    for block in chunkRowBlocks:
                        sImg.extend(block[rowBlockNum])
            sImg = np.clip(np.array(sImg).reshape(row, col), 0, 255).astype(np.uint8)


            if self.oriRow != row or self.oriCol != col:
                return cv2.resize(cv2.merge((sImg, gImg, rImg)), (self.oriCol, self.oriRow))
            else:
                return cv2.merge((sImg, gImg, rImg))
        except Exception as e:
            traceback.print_exc()
            raise e

    # Decode image with text
    def decode_image(self, img):
        try:
            row, col = img.shape[:2]
            if row % 8 != 0 or col % 8 != 0:
                img = self.addPadd(img, row, col)
            row, col = img.shape[:2]
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            bImg, _, _ = cv2.split(img)
            bImg = np.float32(bImg)

            imgBlocks = [bImg[j:j+8, i:i+8] - 128
                         for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
            dctBlocks = [cv2.dct(block) for block in imgBlocks]
            quantizedDCT = [np.round(block / quant) for block in dctBlocks]

            messageBytes = []
            buff = 0
            bitCount = 0
            size_prefix = b''
            message_size = None
            reading_size = True

            for quantizedBlock in quantizedDCT:
                DC = int(quantizedBlock[0][0])
                DC_bits = np.unpackbits(np.array([DC], dtype=np.int8).view(np.uint8))
                buff = (buff << 1) | DC_bits[7]
                bitCount += 1

                if bitCount == 8:
                    byte = buff
                    buff = 0
                    bitCount = 0
                    if reading_size:
                        if byte == ord('*'):
                            try:
                                message_size = int(size_prefix.decode('utf-8'))
                            except:
                                print("Error decoding message size")
                                return ''
                            reading_size = False
                            continue
                        else:
                            size_prefix += bytes([byte])
                    else:
                        messageBytes.append(byte)
                        if len(messageBytes) == message_size:
                            try:
                                return bytes(messageBytes).decode('utf-8')
                            except:
                                return ''

            return ''
        except Exception as e:
            traceback.print_exc()
            raise e

    def toBits(self, byte_data):
        bits = []
        for byte in byte_data:
            binval = bin(byte)[2:].rjust(8, '0')
            bits.append(binval)
        return bits

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def addPadd(self, img, row, col):
        new_row = row + (8 - row % 8) if row % 8 != 0 else row
        new_col = col + (8 - col % 8) if col % 8 != 0 else col
        return cv2.resize(img, (new_col, new_row))

    # Binary File Encode Decode
    def encode_image_file(self, img, secret_bytes):
        try:
            length_prefix = str(len(secret_bytes)).encode('utf-8') + b'*'
            full_bytes = length_prefix + secret_bytes
            self.bitMess = self.toBits(full_bytes)

            row, col = img.shape[:2]
            self.oriRow, self.oriCol = row, col
            if (col // 8) * (row // 8) < len(self.bitMess):
                raise Exception("Error: File too large to encode in image")

            if row % 8 != 0 or col % 8 != 0:
                img = self.addPadd(img, row, col)
            row, col = img.shape[:2]
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            bImg, gImg, rImg = cv2.split(img)
            bImg = np.float32(bImg)

            imgBlocks = [bImg[j:j+8, i:i+8] - 128
                         for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
            dctBlocks = [cv2.dct(block) for block in imgBlocks]
            quantizedDCT = [np.round(block / quant) for block in dctBlocks]

            messIndex = 0
            letterIndex = 0
            for quantizedBlock in quantizedDCT:
                if messIndex >= len(self.bitMess):
                    break
                DC = int(quantizedBlock[0][0])
                DC_bits = np.unpackbits(np.array([DC], dtype=np.int8).view(np.uint8))
                DC_bits[7] = int(self.bitMess[messIndex][letterIndex])
                quantizedBlock[0][0] = float(np.array([np.packbits(DC_bits)[0]], dtype=np.uint8).view(np.int8)[0])
                letterIndex += 1
                if letterIndex == 8:
                    letterIndex = 0
                    messIndex += 1

            sImgBlocks = [cv2.idct(block * quant) + 128 for block in quantizedDCT]
            sImg = []
            for chunkRowBlocks in self.chunks(sImgBlocks, col // 8):
                for rowBlockNum in range(8):
                    for block in chunkRowBlocks:
                        sImg.extend(block[rowBlockNum])
            sImg = np.clip(np.array(sImg).reshape(row, col), 0, 255).astype(np.uint8)

            if self.oriRow != row or self.oriCol != col:
                return cv2.resize(cv2.merge((sImg, gImg, rImg)), (self.oriCol, self.oriRow))
            else:
                return cv2.merge((sImg, gImg, rImg))
        except Exception as e:
            traceback.print_exc()
            raise e

    def decode_image_file(self, img):
        try:
            row, col = img.shape[:2]
            if row % 8 != 0 or col % 8 != 0:
                img = self.addPadd(img, row, col)
            row, col = img.shape[:2]
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            bImg, _, _ = cv2.split(img)
            bImg = np.float32(bImg)

            imgBlocks = [bImg[j:j+8, i:i+8] - 128
                         for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
            dctBlocks = [cv2.dct(block) for block in imgBlocks]
            quantizedDCT = [np.round(block / quant) for block in dctBlocks]

            messageBytes = []
            buff = 0
            bitCount = 0
            size_prefix = b''
            message_size = None
            reading_size = True

            for quantizedBlock in quantizedDCT:
                DC = int(quantizedBlock[0][0])
                DC_bits = np.unpackbits(np.array([DC], dtype=np.int8).view(np.uint8))
                buff = (buff << 1) | DC_bits[7]
                bitCount += 1
                if bitCount == 8:
                    byte = buff
                    buff = 0
                    bitCount = 0
                    if reading_size:
                        if byte == ord('*'):
                            try:
                                message_size = int(size_prefix.decode('utf-8'))
                            except Exception as e:
                                print("Except: ", e)
                                return b''
                            reading_size = False
                            continue
                        else:
                            size_prefix += bytes([byte])
                    else:
                        messageBytes.append(byte)
                        if len(messageBytes) == message_size:
                            return bytes(messageBytes)
            return bytes(messageBytes)
        except Exception as e:
            traceback.print_exc()
            raise e