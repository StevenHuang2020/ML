#python3 Steven
#14/03/2020 download a batch of files use wget 
import os
import wget      #pip instal wget
import argparse

"""""""""""""""""""""""""""""""""""""""""""""""""""""
#usgae:
#python .\downloadFiles.py --ft ./examples/tt.txt --dst ./doc/
"""""""""""""""""""""""""""""""""""""""""""""""""""""

FILESTXT = ''   #files list file
DSTPATH = '.'   #default dest folder

def get_command_line_args():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--ft', type=str, help='Txt file contain files list you want to download')
    parser.add_argument('--dst', type=str, help='Downlaod destination folder')
    
    # Parse and read arguments and assign them to variables if exists
    args, _ = parser.parse_known_args()

    filesText = FILESTXT
    if args.ft:
        filesText = args.ft

    dstPath = DSTPATH
    if args.dst:
        dstPath = args.dst

    return filesText, dstPath

def downloadFile(url,dstPath):
    #print('Beginning file download with wget module')
    fileName = filename = url[url.rfind("/")+1:]
    if dstPath == DSTPATH:
        dst = fileName
    else:
        if dstPath[-1] == '/':
            dst = dstPath + fileName
        else:
            dst = dstPath + "/" + fileName

    print('url=',url)
    print('filename=',filename)
    print('dst=',dst)
    wget.download(url, out=dst)

def main():
    FILESTXT, DSTPATH = get_command_line_args()
    if not os.path.exists(DSTPATH):
        os.mkdir(DSTPATH)

    with open(FILESTXT) as fp:
        for line in fp:
            url = line.rstrip()
            downloadFile(url,DSTPATH)
    
if __name__=='__main__':
    main()
    