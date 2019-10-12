Competition Description£：
Steel is one of the most important building materials of modern times. Steel buildings are resistant to natural and man-made wear which has made the material ubiquitous around the world. To help make production of steel more efficient, this competition will help identify defects.

Severstal is leading the charge in efficient steel mining and production. They believe the future of metallurgy requires development across the economic, ecological, and social aspects of the industry¡ªand they take corporate responsibility seriously. The company recently created the country¡¯s largest industrial data lake, with petabytes of data that were previously discarded. Severstal is now looking to machine learning to improve automation, increase efficiency, and maintain high quality in their production.

The production process of flat sheet steel is especially delicate. From heating and rolling, to drying and cutting, several machines touch flat steel by the time it¡¯s ready to ship. Today, Severstal uses images from high frequency cameras to power a defect detection algorithm.

In this competition, you¡¯ll help engineers improve the algorithm by localizing and classifying surface defects on a steel sheet.

If successful, you¡¯ll help keep manufacturing standards for steel high and enable Severstal to continue their innovation, leading to a stronger, more efficient world all around us.
这个比赛是用于检测钢铁是否存在缺陷，给出的训练数据集中有四类的缺陷情形，当然也存在没有划痕的情形。我先是研读了一份开源的kernel代码，然后对同组的同学的Unet深入的学习的一下，说一下具体思路。训练是采取的两次训练的方法，第一次先是在所有的数据集上进行训练一次，第二次再在全是缺陷的数据集上进行测试；细节方面提一下最后的Loss那里采取的是4个channel的输出，然后对每一个的channel表示一个二分类，采用交叉熵为损失函数。
