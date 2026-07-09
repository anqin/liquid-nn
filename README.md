1. 切换conda到happy-llm
2. 执行：

    $ export HF_ENDPOINT=https://hf-mirror.com
    $ python lfm_train_infer.py


---用阿里云运行leap-finetune后训练--

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
        $ vim mytest/sft_example.yaml   //<--- 修改模型为：LFM2.5-350M

        $ uv run leap-finetune mytest/sft_example.yaml
       
        
