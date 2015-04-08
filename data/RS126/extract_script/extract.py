
def extract_pssm(pssm_file):
    for line in pssm_file:
        print line[:-1]

def extract_dssp(dssp_file):
    return dssp_file.read()

with open('list', 'r') as name_list:
    for line in name_list:
        name = line[:-1]

        with open('dssp/'+name+'.dssp', 'r') as dssp_file:
            dssp = extract_dssp(dssp_file)

        print len(dssp)-1

        with open('pssm/'+name+'.data', 'r') as pssm_file:
            extract_pssm(pssm_file)

        print dssp[:-1]

