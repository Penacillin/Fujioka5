# Forex data scraper

from urllib.parse import urljoin
import shutil
import requests
import zipfile
import os
import csv
from datetime import datetime, timedelta
import numpy as np

dataDir = "../data/"
baseURL = "https://www.truefx.com/dev/data"
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def getFileName(pair, year, month):
    return "{0}-{1}-{2:02d}".format(pair, year, month)

def getZIPURL(year, month, pair):
    monthMap = {1 : "JANUARY", 2 : "FEBRUARY", 3 : "MARCH", 4 : "APRIL", 5 : "MAY" , 6 : "JUNE" , 7 : "JULY" , 8 : "AUGUST", 9 : "SEPTEMBER", 10: "OCTOBER", 11 : "NOVEMBER", 12 : "DECEMBER"}

    directory = "{0}-{1}".format(monthMap[month],year)
    directory2 = "{0}-{1:02d}".format(year, month)

    zipFile = getFileName(pair, year, month) + ".zip"

    url = baseURL + "/" + str(year) + "/" + directory + "/" + zipFile
    url2 = baseURL + "/" + str(year) + "/" + directory2 + "/" + zipFile
    return url, url2, zipFile


def downloadDataForRange(monthStart, yearStart, monthEnd, yearEnd, pair):
    """
    Downloads the zip files and extracts them
    Deletes zip files after

    """
    
    while yearStart < yearEnd or (monthStart <= monthEnd and yearStart <= yearEnd):
        zipURL, zipURL2, fileName = getZIPURL(yearStart, monthStart, pair)
        outputFile = dataDir + fileName
        print("Getting file: ", fileName)
        #print("Checking " , outputFile[0:-4] + ".csv")
        if os.path.exists(outputFile[0:-4] + ".csv"):
            print("File already exists, skipping")
        else:
            print("Downloading...")

            url="https://Hostname/saveReport/file_name.pdf"    #Note: It's https
            r = requests.get(zipURL, verify=False,stream=True)
            if r.status_code != 200:
                r = requests.get(zipURL2, verify=False,stream=True)
            if r.status_code == 200:
                r.raw.decode_content = True

                # write zip file
                with open(outputFile, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                # extract zip file
                with zipfile.ZipFile(outputFile,"r") as zip_ref:
                    zip_ref.extractall(dataDir)
                # Delete zip file
                os.remove(outputFile)
                print("SUCCEEDED")
            else:
                print("FAILED. Status Code: {0} URL: {1}".format(r.status_code, zipURL))

        # Next month
        monthStart += 1
        if monthStart == 13:
            monthStart = 1
            yearStart += 1


def getRawQuotes(dateStart, dateEnd, pair):
    """
    Get raw quotes for range [dateStart, dateEnd] for the pair
    
    :param dateStart: pls
    """
    
    monthStart = dateStart.month
    yearStart = dateStart.year
    monthEnd = dateEnd.month
    yearEnd = dateEnd.year
    downloadDataForRange(monthStart, yearStart, monthEnd, yearEnd, pair)

    data = []
    finished = False

    print("Getting data from ", dateStart , " to " , dateEnd)

    while (yearStart <= yearEnd or (monthStart <= monthEnd and yearStart <= yearEnd)) and not finished:
        with open(dataDir + getFileName(pair, yearStart, monthStart) + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                quoteTime = datetime.strptime(row[1][:-4], "%Y%m%d %H:%M:%S")
                if quoteTime > dateEnd:
                    finished = True
                    break
                if quoteTime > dateStart:
                    data.append([quoteTime, float(row[2]), float(row[3])])
                    # print(quoteTime, float(row[2]), float(row[3]))
                line_count += 1
            print(f'Processed {line_count} lines.')

        # Next month
        monthStart += 1
        if monthStart == 13:
            monthStart = 1
            yearStart += 1

    return data

def getBarData(dateStart, dateEnd, pair, timeFrame):
    """
    Take data and compile it into HLC bars for datetimes [dateStart,dateEnd)

    Returns ndarray with shape (,3) of HLC data

    :param timeFrame: timedelta
    """
    barStart = dateStart

    barData = []
    currentBar = [0,10000,0]
    quotePrice = 0
    while dateStart < dateEnd:
        data = getRawQuotes(dateStart, min(dateStart + timedelta(30), dateEnd), pair)
        print("Got raw quotes " , len(data))
        for quote in data:
            #print("quote: ", quote)
            # Set high if new high
            quotePrice = np.around((quote[1] + quote[2]) / 2, 5)
            if quotePrice > currentBar[0]:
                currentBar[0] = quotePrice
            # Set low if new low
            if  quotePrice < currentBar[1]:
                currentBar[1] = quotePrice
            # close bar
            if quote[0] >= barStart + timeFrame:
                currentBar[2] = quotePrice
                barData.append(currentBar)
                currentBar = [0,10000,0]
                barStart = quote[0]

        dateStart += timedelta(30)

    if currentBar[0] != 0:
        currentBar[2] = np.around(quotePrice, 5)
        barData.append(currentBar)
    
    barData = np.asarray(barData)
    return barData

def getBarDataWithTimestamp(dateStart, dateEnd, pair, timeFrame):
    """
    Take data and compile it into HLC bars for datetimes [dateStart,dateEnd)

    Returns ndarray with shape (,3) of HLC data

    :param timeFrame: timedelta
    """
    barStart = dateStart

    barData = []
    currentBar = [0,0,10000,0]
    quotePrice = 0
    while dateStart < dateEnd:
        data = getRawQuotes(dateStart, min(dateStart + timedelta(30), dateEnd), pair)
        print("Got raw quotes " , len(data))
        for quote in data:
            #print("quote: ", quote)
            # Set high if new high
            quotePrice = np.around((quote[1] + quote[2]) / 2, 5)
            if quotePrice > currentBar[1]:
                currentBar[1] = quotePrice
            # Set low if new low
            if  quotePrice < currentBar[2]:
                currentBar[2] = quotePrice
            # close bar
            if quote[0] >= barStart + timeFrame:
                currentBar[3] = quotePrice
                currentBar[0] = barStart
                barData.append(currentBar)
                currentBar = [0,0,10000,0]
                barStart = quote[0]

        dateStart += timedelta(30)
        print("Have {0} bars".format(len(barData)))

    if currentBar[0] != 0:
        currentBar[3] = np.around(quotePrice, 5)
        currentBar[0] = barStart
        barData.append(currentBar)
    
    barData = np.asarray(barData)
    return barData

def main():
    #getRawQuotes(datetime(2010,1,4), datetime(2010,1,5), "AUDUSD")
    downloadDataForRange(1, 2010, 12, 2018, "AUDUSD")
    
    

if __name__ == "__main__":
    main()