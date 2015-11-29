total=1;
for i = 1: 7
    if(inmodel(i)==1)
        train_new_image_features(:,total)=features_train(:,i);
        total=total+1;
    end
end
