import cv2
import numpy as np
from PIL import Image

class LSB:
    def encode_image(self, origin, embedded_image, output_path):
        """Nhúng ảnh bí mật vào ảnh bìa bằng phương pháp LSB (2 bit)"""
        # Kiểm tra kích thước ảnh bí mật có lớn hơn ảnh bìa không
        if embedded_image.shape[0] > origin.shape[0] or embedded_image.shape[1] > origin.shape[1]:
            raise ValueError("Ảnh bí mật phải nhỏ hơn hoặc bằng kích thước ảnh bìa")

        # Tạo một ảnh trống cùng kích thước với ảnh bìa
        secret_padded = np.zeros_like(origin)
        secret_padded[:embedded_image.shape[0], :embedded_image.shape[1]] = embedded_image

        # Chuyển ảnh bí mật thành dạng nhị phân (2 bit quan trọng nhất)
        secret_bin = np.right_shift(secret_padded, 6)  # Giữ 2 bit quan trọng nhất
        secret_bin = np.left_shift(secret_bin, 6)  # Chuyển về vị trí thấp nhất

        # Xóa 2 bit cuối của ảnh bìa
        cover_bin = np.right_shift(origin, 2)
        cover_bin = np.left_shift(cover_bin, 2)

        # Nhúng ảnh bí mật vào 2 bit thấp nhất của ảnh bìa
        stego = cover_bin | np.right_shift(secret_bin, 6)

        # Lưu ảnh đã nhúng
        stego = cv2.imwrite(output_path, stego)
        print(f"Ảnh đã nhúng được lưu tại: {output_path}")
        return stego

    def decode_image(self,stego, output_path):
        """Giải mã ảnh bí mật từ ảnh đã nhúng"""

        # Lấy 2 bit cuối để tái tạo ảnh bí mật
        secret_bin = np.left_shift(stego & 0b00000011, 6)

        # Lưu ảnh giải mã
        cv2.imwrite(output_path, secret_bin)
        print(f"Ảnh bí mật được giải mã và lưu tại: {output_path}")


    def message_to_binary(self,message):
        return ''.join(format(ord(c), '08b') for c in message)

    def binary_to_message(self, binary_data):
        chars = [chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)]
        return ''.join(chars)

    def encode_message(self, cover_path, output_path, message):
        image = Image.open(cover_path)
        encoded = image.copy()
        binary_msg = self.message_to_binary(message) + '1111111111111110'  # delimiter
        data_index = 0

        pixels = list(encoded.getdata())
        new_pixels = []

        for pixel in pixels:
            if data_index < len(binary_msg):
                new_pixel = []
                for color in pixel[:3]:  # R, G, B
                    if data_index < len(binary_msg):
                        new_color = (color & ~1) | int(binary_msg[data_index])  # set LSB
                        data_index += 1
                    else:
                        new_color = color
                    new_pixel.append(new_color)
                if len(pixel) == 4:
                    new_pixel.append(pixel[3])  # keep alpha if present
                new_pixels.append(tuple(new_pixel))
            else:
                new_pixels.append(pixel)

        encoded.putdata(new_pixels)
        encoded.save(output_path)
        return cv2.imread(output_path)


    def decode_message(self, stego_path):
        image = Image.open(stego_path)
        pixels = list(image.getdata())
        binary_data = ''

        for pixel in pixels:
            for color in pixel[:3]:  # R, G, B
                binary_data += str(color & 1)

        # Look for end delimiter
        end = binary_data.find('1111111111111110')
        if end != -1:
            binary_data = binary_data[:end]
        else:
            print("No message found!")
            return ""

        return self.binary_to_message(binary_data)
