import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('FILE')
config.set('FILE', 'network_file', '2015-05-10 22:46:33.nn')
config.set('FILE', 'training_file', 'data/astral30.pssm')
config.set('FILE', 'validation_file', 'data/casp9.pssm')

config.add_section('MODEL')
config.set('MODEL', 'window_size', '17')
config.set('MODEL', 'hidden_layer_size', '51')

config.add_section('TRAINING')
config.set('TRAINING', 'learning_rate', '1.')
config.set('TRAINING', 'L1_reg', '0.')
config.set('TRAINING', 'L2_reg', '0.')
config.set('TRAINING', 'num_epochs', '1000')
config.set('TRAINING', 'batch_size', '20')

# Writing our configuration file to 'example.cfg'
with open('example.cfg', 'wb') as configfile:
    config.write(configfile)
