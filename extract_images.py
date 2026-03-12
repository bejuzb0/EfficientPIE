from utils.jaad_data import JAAD

if __name__ == '__main__':
    dataset = JAAD(data_path="JAAD")
    dataset.extract_and_save_images()
