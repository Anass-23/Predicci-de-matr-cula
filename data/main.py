# from taumat import CarregaTaules
# from mypp import MyPrettyPrinter
# from mains.taumat2df import StudentsTable
# from bdmat import BDExped

from taumat import CarregaTaules
from mypp import MyPrettyPrinter
from bdmat import BDExped
from pathlib import Path


BASE_PATH     = Path('../../data')
RAW_PATH      = BASE_PATH / 'raw'
INTERIM_PATH  = BASE_PATH / 'interim'

RAW_MAT_FILE_PATH  = RAW_PATH / 'matricules.anon.csv'
RAW_ACRO_FILE_PATH = RAW_PATH / 'acronims.csv'



if __name__ == '__main__':
    pp = MyPrettyPrinter(indent=2)

    (tm, ta) = CarregaTaules(
            nom_mat=RAW_MAT_FILE_PATH,
            nom_acr=RAW_ACRO_FILE_PATH,
            reporta=False
        )
    
    bd = BDExped(tm, ta)
    bd.completa()
    
    # df = bd.get_all_df()
    # df.to_csv(BASE_PATH / 'interim' / 'test.csv', index=False)

    df = bd._get_df_by_idexp('9f474f')
    print(df)

    df.to_csv(INTERIM_PATH / 'test.csv', index=False)
    # df = bd._get_df_by_idexp('08fcfe')
    # print(df)






# if __name__ == '__main__':
#     pp = MyPrettyPrinter(indent=2)

#     (tm, ta) = CarregaTaules(
#             nom_mat='matricules.anon.csv',
#             nom_acr='acronims.csv',
#             reporta=False
#         )

#     # st = StudentsTable(
#     #         tm=tm,
#     #         ta=ta
#     #     )
    
#     bd = BDExped(tm, ta)
#     bd.completa()
#     # df = bd._get_df_by_idexp('9f474f')
#     # print(df)
#     # df = bd._get_df_by_idexp('08fcfe')
#     # print(df)

#     # df.to_csv('test2.csv', index=False)
    
#     df = bd.get_all_df()
#     df.to_csv('ds_base_finalespero.csv', index=False)


    # print('Num mats:', len(bd.bd['9f474f']))
    # print(bd.bd['9f474f'])
    # print(bd.bd['9f474f'].primer_quad())
    # print(bd.bd['9f474f'].ultim_quad())
    # print(bd.bd['9f474f'].durada())

    # idexp : int
    # notae : float
    # viae  : str
    # ordre : int
    # anyn  : int
    # mats  : dict[Quad, BlkMat]

    # idexp = idexp,
    # notae = tm.attr_d_exped(idexp, 'notaacc'),
    # viae  = tm.attr_d_exped(idexp, 'viaacc'),
    # ordre = tm.attr_d_exped(idexp, 'ordreass'),
    # anyn  = tm.attr_d_exped(idexp, 'anynaix'),
    # mats  = {k:d[k] for k in sorted(d)},

    # FUNCTION





    # for a in bd.bd['9f474f'].mats.values():
    #     print(a)
    #     print(a.becat)
    #     print(a.lstmat)
    #     print()


    # pp.pprint(ta.iacro)
    # print(ta.iacro)
    # pp.pprint(tm.expedients['dc0b25'])
    # print(tm.expedients['dc0b25'])
    # pp.pprint(st.st['75745a'])


    # df = st.get_dataframe(ass_target=None)
    # df = st._get_dataframe_exped('75745a', 2)
    # print(df)

    # df.to_csv('tststudents.csv', index=False)