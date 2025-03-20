import streamlit as st
import os


from PIL import Image

from os import listdir
from os.path import isfile, join
from transformers import BlipProcessor, BlipForConditionalGeneration

DATA_PATH = "data"


def predict_caption(image_path):
    try:
        image = Image.open(image_path)
        print(f"image: \nformat: {image.format}, \nsize: {image.size}, \nmode: {image.mode}")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # unconditional image captioning
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

    except Exception as e:
        print(f"Error processing {image}: {e}")

def caption_image(image_file):
    try:
        image = Image.open(image_file)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # unconditional image captioning
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

def compress_image(image):
    with  Image.open(image) as uncompressed_image:
        
        # the original width and height of the image
        image_height = uncompressed_image.height
        image_width = uncompressed_image.width

        print("The original size of Image is: ", round(len(uncompressed_image.fp.read())/1024,2), "KB")

        #compressed the image
        compressed_image = uncompressed_image.resize((image_width,image_height),Image.NEAREST)

        #save the image
        print(f"compressed image name: {image.name}" )
        compressed_image.save(f"{DATA_PATH}/compressed-{image.name}")

        #open the compressed image
        with Image.open(os.path.join(DATA_PATH,f"compressed-{image.name}")) as compressed_image:
            print("The size of compressed image is: ", round(len(compressed_image.fp.read())/1024,2), "KB")
    return compressed_image 

def main():
    uploaded_file = None
    # imageFiles =  [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH,f))]
    st.subheader('Image Caption')
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a Image file",  type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            st.write("filename:", uploaded_file.name)
            
    if uploaded_file is not None:
        caption = f"""<img src="{uploaded_file.name}" alt="{caption_image(uploaded_file)}">"""
        st.image(uploaded_file, caption=caption)
        st.code(caption, language=None)
        with Image.open(uploaded_file) as uncompressed_image:
            st.write("The original size of Image is: ", round(len(uncompressed_image.fp.read())/1024,2), "KB")
            compressed_image = compress_image(uploaded_file)
            with Image.open(compressed_image.filename) as compresed_image:
                st.write("The size of compressed image is: ", round(len(compresed_image.fp.read())/1024,2), "KB")
    #         with open(os.path.join(DATA_PATH,uploaded_file.name), "wb") as f:                
    #             f.write(uploaded_file.getvalue())
    #             st.success("Saved Files")
    #             caption_text=caption_image(f)
    # st.image(uploaded_file, caption=caption_text)
    # for filename in listdir(DATA_PATH):
    #     if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    #         image_path = join(DATA_PATH, filename)
    #         print(f"image path {image_path}")
    #         predict_caption(image_path)

if __name__=="__main__":
    main()

    