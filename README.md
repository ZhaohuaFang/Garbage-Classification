Here, I divide garbage into four categories, which are General waste, Compostable waste, Recyclable waste and Hazardous waste. Instead of classifying garbage into four waste types directly, I classify garbage into sixteen waste-item classes first. For example, if a waste is a light bulb, then it is a hazardous waste. If a given waste is classified as paper, it is a recyclable waste. For implementation, I use classic deep network structures such as VGG-16, ResNet-50 and MobileNet V2 and then compare the output between each network. I found that by using output from waste-item classifier to identify waste types, a higher accuracy could be got in each deep network structure I use.

I was inspired by this paper：[**Municipal Solid Waste Segregation with CNN .pdf**](https://github.com/ZhaohuaFang/Garbage-Classification/blob/main/Municipal%20Solid%20Waste%20Segregation%20with%20CNN%20.pdf)
