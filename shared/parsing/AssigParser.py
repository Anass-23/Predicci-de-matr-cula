import argparse

class AssigParser:
    def __init__(self):
        self.acrlst = []
        self.args   = None
        self.parser = argparse.ArgumentParser(description='Fase 1: Arbre de decisió')
        self.parser.add_argument('-ds', '--dataset', choices=['v1', 'v2'], help='Dataset (v1) sense la motxilla de l\'estudiant i (v2) amb la motxilla', required=True)
        # self.parser.add_argument('-d', '--depth', type=int, help='Profunditat de l\'arbre', required=True)
        
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--all', action='store_true', help='Tots els acrònims')
        group.add_argument('--normal', action='store_true', help='Acrònims habituals per testeig del model')
        group.add_argument('--opt', action='store_true', help='Acrònims de test per optatives')
        # Acrònims de Q2 a Q7
        for quad in range(2, 8):
            group.add_argument(f'--q{quad}', action='store_true', help=f'Acrònims de test (Q{quad})')

    def parse_args(self):
        self.args = self.parser.parse_args()

        if self.args.all: 
            # acrlst = ['MBE', 'F', 'I', 'ISD', 'FMT', 'ES', 'TCO1', 'TP', 'SD', 'TCI', 'MAE', 'TCO2', 'DP', 'EM', 'CSL', 'SA', 'PBN', 'ACO', 'CSR', 'SS', 'PCTR', 'GOP', 'SO', 'XC', 'PDS', 'SEN', 'ESI', 'ASSI', 'SEC', 'IS', 'SAR', 'TFG', 'Q', 'EG', 'CTM', 'SM', 'RM', 'RE', 'GQSIQSMA', 'AE', 'SC', 'SSCI', 'IU', 'MIC', 'BD']
            acrlst = ['MBE', 'F', 'I', 'ISD', 'FMT', 'ES', 'TCO1', 'TP', 'SD', 'TCI', 'MAE', 'TCO2', 'DP', 'EM', 'CSL', 'SA', 'PBN', 'ACO', 'CSR', 'SS', 'PCTR', 'GOP', 'SO', 'XC', 'PDS', 'SEN', 'ESI', 'ASSI', 'SEC', 'IS', 'SAR', 'TFG', 'MIC', 'SC', 'SSCI', 'AE', 'GQSIQSMA', 'BD', 'IU', 'RE']

        else:
            if self.args.q2: 
                acrlst = ['ES', 'TCO1', 'TP', 'SD', 'TCI']
            elif self.args.q3: 
                acrlst = ['MAE', 'TCO2', 'DP', 'EM', 'CSL']
            elif self.args.q4: 
                acrlst = ['SA', 'PBN', 'ACO', 'CSR', 'SS']
            elif self.args.q5: 
                acrlst = ['PCTR', 'GOP', 'SO', 'XC', 'PDS']
            elif self.args.q6: 
                acrlst = ['SEN', 'ESI', 'ASSI', 'SEC']
            elif self.args.q7: 
                acrlst = ['IS', 'SAR']
            elif self.args.normal:
                acrlst = ['MBE', 'F', 'I', 'ISD', 'FMT', 'ES', 'TCO1', 'TP', 'SD', 'TCI', 'MAE', 'TCO2', 'DP', 'EM', 'CSL', 'SA', 'PBN', 'ACO', 'CSR', 'SS', 'PCTR', 'GOP', 'SO', 'XC', 'PDS', 'SEN', 'ESI', 'ASSI', 'SEC', 'IS', 'SAR', 'TFG']
            elif self.args.opt:
                # acrlst = ['Q', 'EG', 'CTM', 'SM', 'RM', 'RE', 'GQSIQSMA', 'AE', 'SC', 'SSCI', 'IU', 'MIC', 'BD']
                acrlst = ['MIC', 'SC', 'SSCI', 'AE', 'GQSIQSMA', 'BD', 'IU', 'RE']
                

        self.acrlst = acrlst