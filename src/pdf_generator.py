from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader  # Make sure to import ImageReader
from PIL import Image
import io
import csv
import os
import subprocess


def insert_image_in_pdf(image_path, pdf_canvas, x, y, width=224, height=224):
    """Insert an image into the PDF canvas."""
    img = Image.open(image_path)
    
    # Resize the image by 50%
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    img = img.resize((new_width, new_height))
    
    # Convert PIL Image to a data stream compatible with reportlab
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    pdf_canvas.drawImage(ImageReader(img_buffer), x, y, new_width, new_height)

def generate_pdf_from_csv(csv_file_name, output_file_name):
    c = canvas.Canvas(output_file_name, pagesize=letter)
    width, height = letter
    
    # Define table headers
    headers = ["S/N", "Image", "Caption", "Match", "Un-match"]
    
    # Starting positions
    x = 50
    y = height - 50  # Start from the top of the page
    row_height = 150  # Adjusted for image height
    col_widths = [30, 180, 180, 80, 80]  # Adjust the column widths as needed
    
    # Draw Table Header
    for idx, header in enumerate(headers):
        c.drawString(x + sum(col_widths[:idx]), y, header)
    
    y -= row_height
    
    # Read data from CSV and fill in the table rows
    with open(csv_file_name, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for sn, row in enumerate(reader, start=1):
            c.drawString(x, y + (row_height - 12) / 2, str(sn))
            
            # Insert image
            insert_image_in_pdf(row["Image"], c, x + col_widths[0], y)
            
            # Text columns
            # Remove the "b'" characters from the start of the caption
            caption = row["Caption"].lstrip("b'")

            # Split caption into multiple lines if it's too long
            caption_lines = [caption[i:i+25] for i in range(0, len(caption), 25)]
            for idx, line in enumerate(caption_lines):
                c.drawString(x + sum(col_widths[:2]), y + (row_height - 12) / 2 - (idx * 12 * 0.8), line)
            
            c.drawString(x + sum(col_widths[:3]), y + (row_height - 12) / 2, row["Match"])
            c.drawString(x + sum(col_widths[:4]), y + (row_height - 12) / 2, row["Un-match"])
            
            y -= row_height
            if y < 100:  # Check for page end
                c.showPage()  # Start a new page
                y = height - 50  # Reset y position
                # Re-draw Table Header on new page
                for idx, header in enumerate(headers):
                    c.drawString(x + sum(col_widths[:idx]), y, header)
                y -= row_height
    
    c.save()
    subprocess.Popen(['xdg-open', output_file_name])

# Example usage, assuming the CSV file is named 'output.csv'
generate_pdf_from_csv("output/Test_sample.csv", "output/Test_sample.pdf")
