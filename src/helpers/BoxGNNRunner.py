from helpers.BaseRunner import *
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from utils.utils import *
import heapq

class BoxGNNRunner(BaseRunner):

    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--test_epoch', type=int, default=-1,
                    help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                    help='Weight decay in optimizer.')
        parser.add_argument('--num_workers', type=int, default=5,
                    help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                    help='pin_memory in DataLoader')
        parser.add_argument('--main_metric', type=str, default='',
                    help='Main metric to determine the best model.')
        parser.add_argument('--check_epoch', type=int, default=10, help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,help='whether to early-stop.')
        parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,help='Dropout probability for each deep layer')
        parser.add_argument('--optimizer', type=str, default='Adam',help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--topk', type=str, default=[10,20,50],help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='["NDCG","HR"]',help='metrics: NDCG, HR')
        parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
        parser.add_argument("--model", default="boxgnn", help="choose in [boxgnn, lfgcf, lightgcn, ngcf, dspr, bpr]")
        parser.add_argument("--num_neg_sample", default=1, type=int)
        parser.add_argument("--pretrain_model_path", default="")
        parser.add_argument("--graph_type", default="all", help="choose in [all, user, item, none]")
        parser.add_argument("--eval_rnd_neg", action="store_true", default=False)
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='embedding size')
        parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
        parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
        parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
        parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
        parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
        parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
        parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
        parser.add_argument('--Ks', nargs='?', default=[10, 20, 50], help='Output sizes of every layer')
        parser.add_argument("--eval_step", default=5, type=int)
        parser.add_argument("--beta", default=1.0, type=float)
        parser.add_argument('--test_flag', nargs='?', default='part',help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
        parser.add_argument("--save",  default=False, action="store_true", help="save the model")
        parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
        parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
        parser.add_argument("--logit_cal", type=str, default="box")

        return parser

    def __init__(self, args):
        self.eval_step=args.eval_step
        self.Ks=args.Ks
        self.test_batch_size=args.test_batch_size
        self.eval_rnd_neg=args.eval_rnd_neg
        self.batch_test_flag=args.batch_test_flag
        self.num_neg_sample=args.num_neg_sample
        self.device=args.device
        super().__init__(args)
        
    def _build_optimizer(self,model):
        logging.info("Optimizer: Adam")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate
        )
        return optimizer
    
    def train(self, dic):
        cur_best_pre_0 = 0  #当前保留的最好指标，通常指recall@10
        stopping_step = 0   #用于记录连续多少个评估没有提升----early stopping计数器
        should_stop = False #是否达到早停阈值
        #保存最优验证结果、对应的测试结果、所在epoch
        best_ret = None     
        best_test_ret = None
        best_epoch = 0

        print("start training ... ")

        model ,corpus= dic['train'].model, dic['train'].corpus

        train_data = torch.LongTensor(corpus.data_gnn['train']).to(self.device)  #构建张量
        train_dataset = corpus.TrainDataset(self,train_data , corpus.data_dict, corpus.data_stat)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        #在训练之前先调用一次，初始度量
        ret = self.test(model, train_dataset, corpus.data_dict["user2item"], corpus.data_dict["val_user2item"], corpus.data_stat)

        optimizer = self._build_optimizer(model)

        for epoch in range(self.epoch):
            #每个epoch调用一次
            train_dataset.create_all_negative_sample()
            model.train()  #进入训练模式
            loss = 0
            train_s_t = time()
            for batch in train_dataloader:
                
                batch_loss, _, _ = model(batch)

                batch_loss = batch_loss
                #反向传播
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss  #累加损失

            """training CF"""

            train_e_t = time()
            #每 eval_stop个epoch进行评估
            if (epoch+1) % self.eval_step == 0 or epoch == 0:
                """testing"""
                model.eval()
                ### valid dataset
                test_s_t = time()

                #先对验证集val_user2item评估 并把结果放入ret，随后调用early_stopping更新cur_best_pre_0,stopping_step
                ret = self.test(model, train_dataset, corpus.data_dict["user2item"], corpus.data_dict["val_user2item"], corpus.data_stat)
                # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], cuda)
                test_e_t = time()

                train_res = PrettyTable()
                train_res.field_names = ["Valid Epoch", "training time", "tesing time", "Loss", "recall", "hit", "ndcg", "precision"]
                # ret = ret[20]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret["hit_ratio"], ret['ndcg'], ret['precision']]
                )
                print(train_res, flush=True)
                
                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            stopping_step, expected_order='acc',
                                                                            flag_step=10)
                #### test dataset
                test_s_t = time()
                test_ret = self.test(model, train_dataset, corpus.data_dict["user2item"], corpus.data_dict["test_user2item"], corpus.data_stat)
                # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], cuda)
                test_e_t = time()

                test_res = PrettyTable()
                test_res.field_names = ["Test Epoch", "training time", "tesing time", "Loss",  "recall", "hit", "ndcg", "precision"]
                # ret = ret[20]
                test_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret["hit_ratio"], test_ret['ndcg'], test_ret['precision']]
                )
                print(test_res, flush=True)
                if should_stop:
                    break

                """save weight"""
                if ret['recall'][0] == cur_best_pre_0:
                    best_epoch = epoch
                    best_ret = ret
                    best_test_ret = test_res
                    # torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint.ckpt"))
            else:
                # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
                print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()), flush=True)
        print(best_test_ret,flush=True)
        print('early stopping at %d, valid recall@10:%.4f, ndcg@10:%.4f' % (best_epoch, cur_best_pre_0, best_ret['ndcg'][0]))
        
    # 生成推荐结果的命中列表  user_pos_test：用户在测试集的正样本 test_items：候选推荐物品列表（包括正样本、负样本）
    # rating：模型对这些物品的预测评分  Ks:用于评估不同的Top-K值
    def ranklist_by_heapq(self,user_pos_test, test_items, rating, Ks):
        item_score = {}  #存储每个候选物品的预测得分
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        # 取出得分最高的前K_max个物品
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        # 将他们与真实点击的user_pos_test做对比，是否命中
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc
    
    #计算推荐列表的AUC（Area Under Curve, Roc曲线下的面积）
    #AUC衡量模型把正样本排在负样本前面的概率
    def get_auc(item_score, user_pos_test):

        #对所有物品按预测得分降序排序
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        #生成排序后的标签序列r
        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        #调用AUC函数计算AUC值
        auc = AUC(ground_truth=r, prediction=posterior)
        return auc

    #将上面两个函数结合起来，同时生成Top-K命中列表和AUC指标
    def ranklist_by_sorted(self,user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def get_performance(self,user_pos_test, r, Ks):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            precision.append(precision_at_k(r, K))
            recall.append(recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(ndcg_at_k(r, K, user_pos_test))
            hit_ratio.append(hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


    def retrieve_all_histories(self,user_ids, n_users, graph, cuda):
        graph = np.array([graph.row, graph.col])
        idx = np.where(np.isin(graph[0], user_ids))[0]
        histories = graph[:, idx]
        return torch.LongTensor(histories.T).to(cuda)

    def test(self,model, dataset, train_user_set, test_user_set, data_stat):
        result = {'precision': np.zeros(len(self.Ks)),
                'recall': np.zeros(len(self.Ks)),
                'ndcg': np.zeros(len(self.Ks)),
                'hit_ratio': np.zeros(len(self.Ks)),
                'auc': 0., 
                "ratings": []}

        n_items = data_stat['n_items']
        n_users = data_stat['n_users']

        # pool = multiprocessing.Pool(cores)

        u_batch_size = self.test_batch_size
        i_batch_size = self.test_batch_size

        test_users = list(test_user_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0
        embs = model.generate()

        user_gcn_emb, entity_gcn_emb = embs
        
        for u_batch_id in tqdm(range(n_user_batchs)):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if self.eval_rnd_neg:
                ### evaluate performance in 100 negative items.
                ### randomly select 100 negative samples 
                negatives = dataset.negative_sampling(user_batch, 100)

                neg_items = torch.LongTensor(negatives).to(self.device)
                pos_items = []
                for u in user_list_batch:
                    pos_items.append(test_user_set[u])
                pos_items = torch.LongTensor(pos_items).to(self.device)

                ### 100个负样本，正样本放在第一个。
                all_items = torch.cat([pos_items, neg_items], dim=-1)
                i_g_embddings = entity_gcn_emb[all_items]
                u_g_embeddings = u_g_embeddings.unsqueeze(1).expand(u_g_embeddings.shape[0], all_items.shape[1], u_g_embeddings.shape[1])
                rate_batch = model.rating(u_g_embeddings, i_g_embddings, same_dim=True).detach()
            else:
                ### evaluate performance among all items
                all_items = torch.LongTensor(np.array(range(0, n_items))).to(self.device)
                if self.batch_test_flag:
                    # batch-item test
                    n_item_batchs = n_items // i_batch_size + 1
                    rate_batch = torch.zeros((len(user_batch), n_items))

                    i_count = 0
                    for i_batch_id in range(n_item_batchs):
                        i_start = i_batch_id * i_batch_size
                        i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                        item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(self.device)
                        i_g_embddings = entity_gcn_emb[item_batch]

                        i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach()

                        rate_batch[:, i_start: i_end] = i_rate_batch
                        i_count += i_rate_batch.shape[1]

                    assert i_count == n_items
                else:
                    # all-item test
                    item_batch = all_items.view(n_items, -1)
                    i_g_embddings = entity_gcn_emb[item_batch]
                    rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach()

            
                for i, u in enumerate(user_list_batch):
                    try:
                        training_items = train_user_set[u]
                        test_items = test_user_set[u]
                    except Exception:
                        training_items = []
                    # user u's items in the test set
                    item_scores =  rate_batch[i][test_items].clone()
                    rate_batch[i][training_items] = -np.inf
                    rate_batch[i][test_items] = item_scores

            maxK = max(self.Ks)
            rate_topK_val, rate_topk_idx = torch.topk(rate_batch, k=maxK, largest=True, dim=-1)
            # user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = []
            for u, rate_topk in zip(user_list_batch, rate_topk_idx):
                r = []
                user_pos_test = test_user_set[u]
                if self.eval_rnd_neg:
                    for idx in rate_topk:
                        ### 100个负样本，正样本放在第一个。
                        if idx.item() == 0:
                            r.append(1)
                        else:
                            r.append(0)
                else:
                    for idx in rate_topk:
                        if idx.item() in user_pos_test:
                            r.append(1)
                        else:
                            r.append(0)
                performance = self.get_performance(user_pos_test, r, self.Ks)

                batch_result.append(performance)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision']/n_test_users
                result['recall'] += re['recall']/n_test_users
                result['ndcg'] += re['ndcg']/n_test_users
                result['hit_ratio'] += re['hit_ratio']/n_test_users
        assert count == n_test_users
        # pool.close()
        return result
    
    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        print("print_res:")
        return 'pass'
    
    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
        model,corpus=dataset.model,dataset.corpus
        predictions=self.test(model, dataset, corpus.data_dict["user2item"], corpus.data_dict["test_user2item"], corpus.data_stat)
        return predictions      #dict