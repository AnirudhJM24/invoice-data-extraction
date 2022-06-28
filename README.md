# invoice-data-extraction
Using ML and DL to extract information from documents mostly invoices on Windows platform.

#Prerequisites
Python 3.4+ installed globally.
Python package venv installed. If not installed run:
```
$ python3 -m pip install virtualenv //or
$ python -m pip install virtualenv
```

## Installing Tessaract OCR:
Install the latest version of (tessaract OCR)[https://github.com/UB-Mannheim/tesseract/wiki] into the C directory and add the path (C:\Program Files (x86)\Tesseract-OCR) to both System and User environment variables in Windows.

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
 python model.py --file [filename relative path]
```
5. Example

```
 python model.py --file examples\airtel_june_2012.pdf
```
6. Results are stored in results.txt and table.csv


