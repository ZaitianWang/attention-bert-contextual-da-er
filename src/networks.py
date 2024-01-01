import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel, AutoConfig

#构建模型

#参数初始化
def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

#当前文本与上下文的attention 加性模型
class Attention(nn.Module):
    def __init__(self,input_dims):
        super().__init__()

        #上下文对应的attention key矩阵
        # self.P_c=nn.Linear(input_dims,1,bias=False) 
        self.P_c=nn.Linear(input_dims,1) 

        #当前上下文对应的attention key矩阵
        # self.P_r=nn.Linear(input_dims,1,bias=False) 
        self.P_r=nn.Linear(input_dims,1) 

        #偏差项
        # self.b=torch.tensor([[np.random.random()]],requires_grad=True).cuda()

        #激活函数
        self.relu=nn.ReLU()

        #attention归一化
        self.softmax=nn.Softmax(dim=-1)


    def forward(self,contexts,current):

        # O=self.relu(self.P_c(contexts)+self.P_r(current)+self.b)
        O=self.relu(self.P_c(contexts)+self.P_r(current))
        A=self.softmax(O)
    
        return A



#bert+每个情感对应一个attention
class BERTAttentionsModel_v1(nn.Module):
    def __init__(self, n_classes, model_name):
        super(BERTAttentionsModel_v1, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.base = BertModel.from_pretrained(model_name, config=config)

        self.dim = 1024 if 'large' in model_name else 768 #此时为768
        dim=self.dim

        self.attention = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        #不同情感使用不同attention
        self.love_attention=Attention(dim)
        self.joy_attention=Attention(dim)
        self.fright_attention=Attention(dim)
        self.anger_attention=Attention(dim)
        self.fear_attention=Attention(dim)
        self.sorrow_attention=Attention(dim)


        #attention后接全连接层
        self.out_love = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_joy = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fright = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_anger = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fear = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_sorrow = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

        init_params([self.out_love, self.out_joy, self.out_fright, self.out_anger,
                     self.out_fear,  self.out_sorrow, self.attention])

    def forward(self,input_tokens,attention_masks):
        #所有文本bert输出的隐藏层  
        shape=input_tokens.shape
        input_tokens=input_tokens.reshape(-1,shape[-1])
        attention_masks=attention_masks.reshape(-1,shape[-1])
        roberta_output = self.base(input_ids=input_tokens,attention_mask=attention_masks)
        # print(roberta_output)
        # print(roberta_output.last_hidden_state.shape)
        last_layer_hidden_states = roberta_output.hidden_states[-1] 
        
        # shape 4,7, 128, 768
        last_layer_hidden_states=last_layer_hidden_states.reshape(shape[0],7, 128, 768)
        # print(f"last_layer_hidden_states 4,7, 128, 768 {last_layer_hidden_states.shape}")

        # shape 4 7 128 1
        weights = self.attention(last_layer_hidden_states)
        # print(f"weights 4 7 128 1 {weights.shape}")
 
        #隐藏层向量自注意力后的向量 shape 4 7 768
        vector = torch.sum(weights*last_layer_hidden_states, dim=2)#shape(batchsize/gpu_num,768)
        # print(f"vector 4 7 768 {vector.shape}")
    
        #当前句子隐藏层权重 
        current_vector=vector[:,0,:] #shape 4  768
        current_vector_reshape=current_vector.reshape(shape[0],1,768) #shape 4 1 768
        # print(f"current_vector 4 1 768 {current_vector.shape}")

        #上下文隐藏层权重 shape 4 6 768
        context_vector=vector[:,1:,:]

        #每种情感分别attention 得到权重 shape 4 6 1
        love_weights=self.love_attention(current_vector_reshape,context_vector)
        joy_weights=self.joy_attention(current_vector_reshape,context_vector)
        fright_weights=self.fright_attention(current_vector_reshape,context_vector)
        anger_weights=self.anger_attention(current_vector_reshape,context_vector)
        fear_weights=self.fear_attention(current_vector_reshape,context_vector)
        sorrow_weights=self.sorrow_attention(current_vector_reshape,context_vector)
        # print(f"love_weights 4 6 1 {love_weights.shape}")

        #按照权重加权平均 shape 4 768
        love_vector=torch.sum(love_weights*context_vector, dim=1)
        joy_vector=torch.sum(joy_weights*context_vector, dim=1)
        fright_vector=torch.sum(fright_weights*context_vector, dim=1)
        anger_vector=torch.sum(anger_weights*context_vector, dim=1)
        fear_vector=torch.sum(fear_weights*context_vector, dim=1)
        sorrow_vector=torch.sum(sorrow_weights*context_vector, dim=1)
        # print(f"love_vector 4 768 {love_vector.shape}")

        #加上当前句子权重

        # print(f"current_vector 4 768 {current_vector.shape}")
        love_vector+=current_vector
        joy_vector+=current_vector
        fright_vector+=current_vector
        anger_vector+=current_vector
        fear_vector+=current_vector
        sorrow_vector+=current_vector

        #attention后过全连接
        love = self.out_love(love_vector)
        joy = self.out_joy(joy_vector)
        fright = self.out_fright(fright_vector)
        anger = self.out_anger(anger_vector)
        fear = self.out_fear(fear_vector)
        sorrow = self.out_sorrow(sorrow_vector)  

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
    

class IQIYModelLite(nn.Module):
    def __init__(self, n_classes, model_name):
        super(IQIYModelLite, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.base = BertModel.from_pretrained(model_name, config=config)

        dim = 1024 if 'large' in model_name else 768

        self.attention = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        # self.attention = AttentionHead(h_size=dim, hidden_dim=512, w_drop=0.0, v_drop=0.0)

        self.out_love = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_joy = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fright = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_anger = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fear = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_sorrow = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

        init_params([self.out_love, self.out_joy, self.out_fright, self.out_anger,
                     self.out_fear,  self.out_sorrow, self.attention])

    def forward(self, input_ids, attention_mask):
        roberta_output = self.base(input_ids=input_ids,
                                   attention_mask=attention_mask)

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        # print(weights.size())
        context_vector = torch.sum(weights*last_layer_hidden_states, dim=1)
        # context_vector = weights

        love = self.out_love(context_vector)
        joy = self.out_joy(context_vector)
        fright = self.out_fright(context_vector)
        anger = self.out_anger(context_vector)
        fear = self.out_fear(context_vector)
        sorrow = self.out_sorrow(context_vector)

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
    
