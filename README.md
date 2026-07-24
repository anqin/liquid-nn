#切换conda到happy-llm
#执行：
#$ export HF_ENDPOINT=https://hf-mirror.com
#$ python lfm_train_infer.py

1. 申请低配CPU高带宽机器，修改密码并登录

    1.1 创建账号并准备环境

        # useradd -m -s /bin/bash anqin
        # passwd anqin
        # su anqin

        $ git clone <github/leap-finetune>
    
    1.2 安装uv并启动uv sync同步数据

        $ cd leap-finetune
        $ curl -LsSf https://astral.sh/uv/install.sh | sh
        $ source ~/.bashrc

        $ uv sync  //<---这步骤耗时一天，一般用nohup启动
        $ nohup uv sync > nohup_run.log 2>&1 &  //<--可以通过top -H -p $(pgrep -f "uv")看运行情况

    1.3 下载完成后，创建快照

2. 申请GPU高配机器，修改密码并登录

    2.1 创建并挂着云盘，选择从快照中创建

    2.2 创建账号并准备环境

        # useradd -m -s /bin/bash anqin
        # passwd anqin
        
        # mount /dev/sdb3  /mnt

        # su anqin
        $ cd 
        $ cp /mnt/home/anqin/leap-finetune  . -rf
        $ cd leap-finetune
        $ uv sync    //<--- 检查依赖是否有问题

    2.3 设置环境变量并登录huggingface

        $ export HF_ENDPOINT=https://hf-mirror.com
        $ uv run hf auth login   //<-- 这里需要输入access token，在hf_token.txt中

    2.4 运行SFT

        $ mkdir mytest
        $ cp job_config/sft_example.yaml  mytest
        $ vim mytest/sft_example.yaml   //<--- 修改模型为：LFM2.5-350M，会自动从HF下载到~/.cache下面

        $ uv run leap-finetune mytest/sft_example.yaml
       
    2.5 模型打包

        $ uv tool install leap-bundle   //<-- 安装leap-bundle, 用于上传/下载HF上模型（本地运行用不上）
        
        # 由于leap-finetune的SFT训练出来的是LORA（目录下没有config.json，只有adapter_config.json），所以要merge
        # (1) 查找基座模型路径，如果没有就通过leap-bundle下载
        # (2) 修改merge_model_lora.py的路径，填写基座模型路径和Adapter模型（LORA）路径
        # (3) 运行脚本合并成运行的HF格式模型，并填写output路径

        $ ls ~/.cache/huggingface/hub/... //<--- 基座模型在本地路径大致位置
        $ uv run python merge_model_lora.py  //<-- 由于leap-finetune的SFT训练出来的是LORA（目录下没有config.json，只有adapter_config.json），所以要merge
        

    2.6 对新模型运行测试

        $ vim mytest/eval_standalone_example.yaml

        在model_name下面增加：

        checkpoint: "outputs/my_complete_hf_model" //<--- 指向 merge_model_lora.py的output路径

        然后，运行测试命令：

        $ uv run leap-finetune eval mytest/eval_standalone_example.yaml 


    2.7 运行web版对话

        # 安装streamlit
        $ uv pip install streamlit

        # 运行app启动web
        $ uv run streamlit dialog_app_web.py

    2.8 运行CLI版对话

        $ uv run python dialog_cli.py


3. 其他

    3.1 HF模型下载（从HF hub下载指定模型）

        $ export HF_ENDPOINT=https://hf-mirror.com
        $ uv run hf auth login

        $ uv run hf download LiquidAI/LFM2.5-8B-A1B
