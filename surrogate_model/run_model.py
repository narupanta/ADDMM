from core.utils import * 
from core.datasetclass import LinearElasticityDataset
from core.model_graphnet import EncodeProcessDecode
from tqdm import tqdm
from torch_geometric.loader import DataLoader

if __name__ == "__main__" :
    torch.manual_seed(42) 

    output_dir = "/home/narupanta/ADDMM/surrogate_model/output"
    run_dir = prepare_directories(output_dir)
    model_dir = os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LinearElasticityDataset(data_dir = "/home/narupanta/ADDMM/surrogate_model/dataset")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle = True)
    model = EncodeProcessDecode(node_feat_size=4,
                                edge_feat_size= 1,
                                output_size=2,
                                latent_size=128,
                                message_passing_steps=15)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)

    # Training loop
    num_epochs = 100
    train_loss_per_epochs = []
    model.to(device)
    is_accumulate_normalizer_phase = True
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for idx_traj, batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = model.loss_function(predictions, batch.y)

            # Backpropagation
            if is_accumulate_normalizer_phase is False : # use first epoch to accumulate normalizer
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
                loop.set_description(f"Trajectory {idx_traj + 1}/{len(train_loader)}")
                loop.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Trajectory {idx_traj + 1}/{len(train_loader)}, MSE Loss {loss.item():.4f}")

        if is_accumulate_normalizer_phase is False :
            avg_train_loss = train_total_loss / len(train_loader)
            train_loss_per_epochs.append(avg_train_loss)
            model.save_model(model_dir)
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
        else :
            is_accumulate_normalizer_phase = False