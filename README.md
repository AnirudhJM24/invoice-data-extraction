# invoice-data-extraction
Using ML and DL to extract information from documents mostly invoices on Windows platform.

# Prerequisites
Python 3.4+ installed globally.
Python package venv installed. If not installed run:
```
$ python3 -m pip install virtualenv //or
$ python -m pip install virtualenv
```

## Installing Tessaract OCR:
Install the latest version of [tessaract OCR](https://github.com/UB-Mannheim/tesseract/wiki) into the C directory and add the path (C:\Program Files \Tesseract-OCR) to both System and User environment variables in Windows. Download the additional eng_layer.traineddata file and add it to C:\Program Files\Tesseract-OCR\tessdata

## Running the Code.
1. Clone the repository or downlaod the zip file from GitHub
```
 git clone https://github.com/abhayhk2001/invoice-data-extraction
```

2. Open a Terminal window in the same folder as the downloaded code.
```
 pip install -r requirements.txt
```

3. Add the invoice to examples subfolder.

4. 
```
 python main.py --file [filename relative path]
```
5. Example

```
 python main.py --file examples\airtel_june_2012.pdf
```
6. Results are stored in results.txt and table.csv

## Retraining
The images provided to us are present in example and they have already been trained upon.
Any new images to be trained can be added to the annotated folder and then annotated by using Labelimg.

### Add Images to the annotated/ folder in the project
### Annotate Images using Labelimg

1. Instructions for annotating is given in the link below:
2. It also has a video showing an example

[Instructions](https://drive.google.com/drive/u/0/folders/1CoCIzraThqebXwIsk-WVtj2z7V_BgpV_)

If the new invoice is a pdf convert to images and add to annotated/
If the new invoice has multiple pages add each page as a different image to annotated/ and annotate.

After these steps run the following command to retrain.
```
 python yolo_trainer.py
```