def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--episodes', type=int, default=1000000, help='number of episodes')

    # environment hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--buffer_start', type=int, default=5000, help='buffer start size')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='learning rate for training')
    parser.add_argument('--max_buffer_size', type=int, default=10000, help='maximum buffer size')

    # network hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon for epsilon-greedy')
    parser.add_argument('--epsilon_decay_steps', type=float, default=100000, help='epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.025, help='minimum epsilon')

    # save, write, print frequency
    parser.add_argument('--update_target_net_freq', type=int, default=5000, help='update target network frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--write_freq', type=int, default=100, help='write frequency')
    parser.add_argument('--print_freq', type=int, default=1000, help='print frequency')


    #prioritized queue hyperparameters
    parser.add_argument('--prioritized_alpha', type=float, default=0.6, help='alpha for prioritized experience replay')
    parser.add_argument('--prioritized_beta', type=float, default=0.4, help='beta for prioritized experience replay')
    parser.add_argument('--prioritized_beta_increment', type=float, default=0.001, help='beta increment for prioritized experience replay')

    #multi-step learning hyperparameters
    parser.add_argument('--n_step', type=int, default=3, help='n-step learning')

    #specify locations for saving
    parser.add_argument('--data_dir', type=str, default='data/', help='directory to save data')
    parser.add_argument('--model_name', type=str, default='model.pth', help='name of model')
  


    #A3C hyperparameters

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_steps', type=int, default=1000000, help='maximum number of steps')
    parser.add_argument('--feature_size', type=int, default=512, help='feature size')
    



    return parser

# linear decay (epsiolon -epsilon_min) / epsilon step