import argparse


class BaseConfig():
    
    def parse(self):
        parser = argparse.ArgumentParser()
        for k, v in vars(self).items():
            if not k.startswith('__'):
                if v is not None:
                    if isinstance(v, bool):
                        if v:
                            parser.add_argument(f'--no_{k}', action='store_false')
                        else:
                            parser.add_argument(f'--{k}', action='store_true')
                    else:
                        parser.add_argument(f'--{k}', type=type(v), default=v)
                else:
                    parser.add_argument(f'--{k}')
                setattr(self, k, v)
        args = parser.parse_args()
        for k, v in vars(args).items():
            setattr(self, k, v)
            

            
    
            
        
            