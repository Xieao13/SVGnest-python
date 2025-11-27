import wandb

# 只需要初始化一个离线运行，它会自动退出
# 它的唯一目的就是创建一个干净的、带有新ID的文件夹
run = wandb.init(
    project="ppo_svgnest",
    name="placeholder_for_recovery",
    mode="offline"  # 非常重要：使用离线模式
)

# 打印出这个新运行的文件夹路径，方便我们下一步找到它
print(f"\n成功创建占位文件夹: {run.dir}")

run.finish()