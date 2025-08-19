import easyocr
from transformers import TrOCRProcessor
from optimum.intel.openvino import OVModelForVision2Seq
from PIL import Image

MODEL_DIR='./ocr_pipeline/models/trocr'


class OCRPipeline:
    def __init__(self):
        self.recognized_lines = []
        self.processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        self.model = OVModelForVision2Seq.from_pretrained(MODEL_DIR)
        self.reader = easyocr.Reader(['en']) 

    def load_image(self, IMAGE_DIR):
        """
        Loads the image from a directory into a pillow Image object
        """
        return Image.open(IMAGE_DIR).convert("RGB")
    
    def bound_text(self, IMAGE_DIR):
        """
        Takes in an image directory and returns the bounding boxes for the text lines
        """
        detection_results = self.reader.readtext(IMAGE_DIR)
        bounding_boxes = [item[0] for item in detection_results]

        return bounding_boxes

    def crop_images(self, bounding_boxes, image):
        """
        Uses the bounding boxes of the text lines from an original image and outputs the cropped boxes
        """
        images = []
        for box in bounding_boxes:
            left = min([point[0] for point in box])
            top = min([point[1] for point in box])
            right = max([point[0] for point in box])
            bottom = max([point[1] for point in box])
            cropped_image = image.crop((left, top, right, bottom))
            images.append(cropped_image)
        
        return images
    
    def image_to_text(self, images):
        """
        Transforms the cropped images to text
        """
        text = []
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text.append(generated_text)

        return text

    def inference(self, IMAGE_DIR):
        """
        Collectively performs the image handling and inferencing to output the text from an image
        """
        img = self.load_image(IMAGE_DIR)
        bounding_boxes = self.bound_text(IMAGE_DIR)
        cropped_images = self.crop_images(bounding_boxes, img)
        text = self.image_to_text(cropped_images)
        final_text = " ".join(text)

        return final_text

        



        
