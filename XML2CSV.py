import xml.etree.ElementTree as ET
import pandas as pd

# xml_data = 'iot.stackexchange/Posts.xml'
xml_data = 'stats.stackexchange.com/Posts.xml'

def xml2df(xml_data):
    tree = ET.parse(xml_data)
    root = tree.getroot()
    allRecords = []
    for child in root:
        allRecords.append(child.attrib)
    return pd.DataFrame(allRecords)

data = xml2df(xml_data)
data.to_csv('stats_posts.csv')
