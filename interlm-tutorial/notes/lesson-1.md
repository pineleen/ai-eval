上海AI LAB， interlm， 书生-浦语

* 开源数据集： 万卷
* 预训练框架： interlm train
* 微调框架： xtuner， 全参数finetune（垂类数据续训）， lora等部分参数微调
* 部署：lmdeploy
* 评测： open compase
* 应用开发： lagent，智能体


基础预训练模型，在训练完之后并不能直接使用，必须经过finetune，才能获取到固定的回答模式。
