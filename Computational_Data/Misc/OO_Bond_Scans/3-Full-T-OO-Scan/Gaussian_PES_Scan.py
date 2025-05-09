#!/usr/bin/python
import sys

def getrawdata(infile):
        f=open(infile,'r')
        opt=0
        geo=0
        struct=[]
        structure=[]
        energies=[]
        energy=[]
        for line in f:
                
                if opt==1 and geo==1 and not ("---" in line):
                        structure+=[line.rstrip()]
                
                if 'Coordinates (Angstroms)' in line:
                        if opt==0:
                                opt=1
                                structure=[]
                        
                if opt==1 and "--------------------------" in line:
                        if geo==0:
                                geo=1
                        elif geo==1:
                                geo=0
                                opt=0
                if 'SCF Done' in line:
                        energy=filter(None,line.rstrip('\n').split(' '))
                if      'Optimization completed' in line and (opt==0 and geo==0):
                        energies+=[float(energy[4])]
                        opt=0
                        geo=0
                        struct+=[structure]
                        structure=[]
        
        return struct, energies

def periodictable(elementnumber):
        ptable={1:'H',2:'He',\
        3:'Li', 4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',\
        11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',\
        19:'K',20:'Ca',\
        21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',\
        31:'Ga',32:'Ge',33:'As',34:'Se',35:'Br',36:'Kr',\
        37:'Rb',38:'Sr',\
        39:'Y',40:'Zr',41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',\
        49:'In',50:'Sn',51:'Sb',52:'Te',53:'I',54:'Xe',\
        55:'Cs',56:'Ba',\
        57:'La',58:'Ce',59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',66:'Dy',67:'Ho',68:'Er',69:'Tm',70:'Yb',71:'Lu',\
        72:'Hf', 73:'Ta', 74:'W',75:'Re', 76:'Os', 77:'Ir',78:'Pt', 79:'Au', 80:'Hg',\
        81:'Tl', 82:'Pb', 83:'Bi',84:'Po',85:'At',86:'Rn',\
        87:'Fr',88:'Ra',\
        89:'Ac',90:'Th',91:'Pa',92:'U',93:'Np',94:'Pu',95:'Am',96:'Cm',97:'Bk',98:'Cf',99:'Es',100:'Fm',101:'Md',102:'No',\
        103:'Lr',104:'Rf',105:'Db',106:'Sg',107:'Bh',108:'Hs',109:'Mt',110:'Ds',111:'Rg',112:'Cn',\
        113:'Uut',114:'Fl',115:'Uup',116:'Lv',117:'Uus',118:'Uuo'}
        element=ptable[elementnumber]
        return element

def genxyzstring(coords,elementnumber):
        x_str='%10.5f'% coords[0]
        y_str='%10.5f'% coords[1]
        z_str='%10.5f'% coords[2]
        element=periodictable(int(elementnumber))
        xyz_string=element+(3-len(element))*' '+10*' '+\
        (8-len(x_str))*' '+x_str+10*' '+(8-len(y_str))*' '+y_str+10*' '+(8-len(z_str))*' '+z_str+'\n'
 
        return xyz_string

def getstructures(rawdata):
        
        n=0
        for structure in rawdata:
                
                n=n+1
                num="%03d" % (n,)
                g=open('structure_'+num+'.xyz','w')
                itson=False
                cartesian=[]
                        
                for item in structure:
                        
                        coords=filter(None,item.split(' '))
                        coordinates=[float(coords[3]),float(coords[4]),float(coords[5])]
                        element=coords[1]
                        cartesian+=[genxyzstring(coordinates,element)]
                g.write(str(len(cartesian))+'\n')
                g.write('Structure '+str(n)+'\n')
                for line in cartesian:
                        g.write(line)
                g.close()
                cartesian=[]
        return 0
        
if __name__ == "__main__":
        infile=sys.argv[1]
        rawdata,energies=getrawdata(infile)
        structures=getstructures(rawdata)
        g=open('energies.dat','w')
        for n in range(0,len(energies)):
                g.write(str(n)+'\t'+str(energies[n])+'\n')
        g.close()