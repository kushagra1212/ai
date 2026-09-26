import struct


def load_mnist_raw(images_path, labels_path, max_samples=3):
  with open(images_path, 'rb') as f_img, open(labels_path, 'rb') as f_lbl:
    img_magic, num_images, num_rows, num_cols = struct.unpack(
        '>IIII', f_img.read(16)
    )
    lbl_magic, num_labels = struct.unpack('>II', f_lbl.read(8))

    print('MNIST Header Info:')
    print(f'  Images Magic : {img_magic} (Expected 2051)')
    print(f'  Total Images : {num_images}')
    print(f'  Dimensions   : {num_rows} x {num_cols} pixels')
    print(f'  Total Labels : {num_labels}\n')

    samples = []
    count = min(max_samples, num_images)
    for _ in range(count):
      lbl = struct.unpack('B', f_lbl.read(1))[0]
      raw_pixels = f_img.read(num_rows * num_cols)
      pixels = [
          [
              raw_pixels[r * num_cols + c] / 255.0
              for c in range(num_cols)
          ]
          for r in range(num_rows)
      ]
      samples.append({'label': lbl, 'pixels': pixels})

  return samples


def print_mnist_ascii(sample, sample_idx):
  print(f"Sample #{sample_idx} | True Label: [{sample['label']}]")
  print('-' * 56)
  for r in range(28):
    line = '  '
    for c in range(28):
      p = sample['pixels'][r][c]
      if p > 0.75:
        line += '##'
      elif p > 0.40:
        line += '**'
      elif p > 0.10:
        line += '..'
      else:
        line += '  '
    print(line)
  print()


if __name__ == '__main__':
  samples = load_mnist_raw(
      'data/mnist/train-images-idx3-ubyte',
      'data/mnist/train-labels-idx1-ubyte',
      max_samples=3,
  )
  for i, s in enumerate(samples):
    print_mnist_ascii(s, i)
