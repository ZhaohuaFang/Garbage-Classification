#建立两个list，记录对应的真值和预测值
predd=[]
true=[]
for x,y in db_test:
  
  out=newnet(x)
  pred = tf.argmax(out, axis=1)
  pred = tf.cast(pred, dtype=tf.int32)
  #print(pred)
  pred=pred.numpy()
  pred=list(pred)

  predd=predd+pred

  y=y.numpy()
  y=list(y)

  true=true+y  
#建立混淆矩阵
c=confusion_matrix(true,predd,labels=[i for i in range(20)])

#矩阵分割，根据具体问题定义新的混淆矩阵，可参考论文
c1=c[0:5,:]
c2=c[5:10,:]
c3=c[10:15,:]
c4=c[15:20,:]

food1=c1[:,0:5]
food2=c1[:,5:10]
food3=c1[:,10:15]
food4=c1[:,15:20]

recycle1=c2[:,0:5]
recycle2=c2[:,5:10]
recycle3=c2[:,10:15]
recycle4=c2[:,15:20]

other1=c3[:,0:5]
other2=c3[:,5:10]
other3=c3[:,10:15]
other4=c3[:,15:20]

bad1=c4[:,0:5]
bad2=c4[:,5:10]
bad3=c4[:,10:15]
bad4=c4[:,15:20]


food1=food1.sum()
food2=food2.sum()
food3=food3.sum()
food4=food4.sum()
recycle1=recycle1.sum()
recycle2=recycle2.sum()
recycle3=recycle3.sum()
recycle4=recycle4.sum()
other1=other1.sum()
other2=other2.sum()
other3=other3.sum()
other4=other4.sum()
bad1=bad1.sum()
bad2=bad2.sum()
bad3=bad3.sum()
bad4=bad4.sum()

acc1=food1/(food1+food2+food3+food4)
acc2=recycle2/(recycle1+recycle2+recycle3+recycle4)
acc3=other3/(other1+other2+other3+other4)
acc4=bad4/(bad1+bad2+bad3+bad4)
print((acc1+acc2+acc3+acc4)/4)
