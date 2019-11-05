# nrn
Codes and materials for paper *Mapping human activity volumes through remote sensing imagery*.

The proposed network, called Neighbor-ResNet, is a modified deep residual network taking spatial autocorrelation into account. 

**Data:**

**Input data:** 

​	 <u>remote sensing imagery collected from open-sourced Google Maps</u>

   （<https://github.com/linuxexp/Google-Maps-Downloader>）

​	Spatial resolution: 19.1m

​	Extent: 0.03° for image parcels, with the middle 0.01° area as target estimation area

​	Project: WGS84 Web Mercator

**Output data:**

​	 <u>positioning  data collected from Tencent applications</u>

​	 Due to the privacy concern, we only share the data information and attributions.

​	 Date: 18 January 2016 to 22 January 2016

​	 Spatial resolution: 0.01° in latitude × 0.01° in longtitude

| Id   | min_lat | min_lon | volumes |
| ---- | ------- | ------- | ------- |
| 1    | 123.16  | 41.70   | xxxxxx  |
| 2    | 123.16  | 41.71   | xxxxxx  |
| …    | …       | …       | …       |

