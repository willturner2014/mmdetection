import os
import pandas as pd
import numpy as np
import cv2
import csv
from geojson import FeatureCollection,Feature,MultiPolygon,Polygon
from osgeo import gdal

feat_col=FeatureCollection([])

#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset

#获取仿射矩阵信息
def Getgeotrans(fileName):
    dataset = readTif(fileName)
    return dataset.GetGeoTransform()

'''
0：图像左上角的X坐标；
1：图像东西方向分辨率；
2：旋转角度，如果图像北方朝上，该值为0；
3：图像左上角的Y坐标；
4：旋转角度，如果图像北方朝上，该值为0；
5：图像南北方向分辨率；
'''
#像素坐标和地理坐标仿射变换
def CoordTransf(pixel_coor,GeoTransform):
    XGeo = GeoTransform[0]+GeoTransform[1]*pixel_coor[0]+pixel_coor[1]*GeoTransform[2];
    YGeo = GeoTransform[3]+GeoTransform[4]*pixel_coor[0]+pixel_coor[1]*GeoTransform[5];
    return [XGeo,YGeo]

def Pixels2Geo(pixel_coor,GeoTransform):
    
    geo_coords=[]
    # for pixel_coor in pixel_coors:
    XGeo1 = GeoTransform[0]+GeoTransform[1]*pixel_coor[0]+pixel_coor[1]*GeoTransform[2];
    YGeo1 = GeoTransform[3]+GeoTransform[4]*pixel_coor[0]+pixel_coor[1]*GeoTransform[5];
    XGeo2 = GeoTransform[0]+GeoTransform[1]*pixel_coor[2]+pixel_coor[3]*GeoTransform[2];
    YGeo2 = GeoTransform[3]+GeoTransform[4]*pixel_coor[2]+pixel_coor[3]*GeoTransform[5];
    # [XGeo,YGeo]=CoordTransf(pixel_coor,GeoTransform)
    geo_coords.append([XGeo1,YGeo1])
    geo_coords.append([XGeo2,YGeo2])
    geo_coords = [[XGeo1,YGeo1],[XGeo1,YGeo2],[XGeo2,YGeo2],[XGeo2,YGeo1],[XGeo1,YGeo1]]

    return geo_coords



def get_files_by_suffix(path, suffix):
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(suffix):
            full_path = os.path.join(path, file_name)
            files.append([full_path, file_name])
    return files


def main():    # 主函数的代码
    csv_path=r'D:\Codes\Python\AI\4_Baseline\openmmlab\mmyolo-0.5.0\demo\vis\houhai-res\yolov5_0705\0.05\03'
    image = cv2.imread(r'D:\Codes\Python\AI\4_Baseline\openmmlab\mmyolo-0.5.0\demo\vis\houhai-res\yolov5_0705\0.05\201912.png')
    blank = np.zeros(image.shape, np.uint32) 
    csv_file=get_files_by_suffix(csv_path, 'csv')
    csv_result_path=''
    transform =Getgeotrans(r'D:\Codes\Python\AI\4_Baseline\openmmlab\mmyolo-0.5.0\demo\houhai\01\20221224.png')
    

    for i in range(len(csv_file)):
    #   file =pd.read_csv(csv_path+str(csv_file[i]),engine='python')
        csv_in_file =pd.read_csv(csv_file[i][0],engine='python')
        first_column_values = csv_in_file.iloc[:, 0].tolist()
        csv_result_path=csv_path+"\\result\\"+csv_file[i][1]+'_score.csv'
        with open(csv_result_path, mode='w', newline='') as csv_out_file:
            writer = csv.writer(csv_out_file)
            for value in first_column_values:
                try:
                    global feat_col
                    coords = value.split('[[')[1].split(']]')[0].split(',')
                    coords = list(map(lambda x: float(x), coords))
                    geo_coords = Pixels2Geo(coords,transform )
                    score =value.split('scores: tensor([')[1].split(']')[0]
                    score = float(score)
                    coords.append(score)
                    # out_str=",".join(coords)#这样数字会打散
                    writer.writerow(coords)
                    # poly_coords=geo_coords
                    # poly_coords.append(geo_coords[0])
                    poly=Polygon([geo_coords])
                    feat=Feature(geometry=poly, properties={"time": csv_file[i][1].split('.')[0],"score": score})
                    feat_col.features.append(feat)
                except Exception as e:
                    print(e)
                    # print("err-1",csv_file[i][1],value)
                    print("err-1",csv_file[i][1])

    # print(feat_col)
    out_geojson=csv_path+"\\result\\"+csv_file[i][1]+'_geo.json'
    with open(out_geojson, mode='w+', newline='') as out_geojson_file:
        # writer = csv.writer(out_geojson_file)#这样有逗号分隔
        # writer.writerow(str(feat_col))
        out_geojson_file.write(str(feat_col))

if __name__ == "__main__":
        main()


