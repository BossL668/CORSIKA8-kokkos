#!/usr/bin/env python3

# (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.

import pickle, sys, itertools



def load_particledb(filename):
    '''
    loads the pickled particle_db (which is an OrderedDict)
    definition of particle_db dict is: "name", "antiName", "pdg", "mass", "charge", "lifetime", "ngc_code", "isNucleus", "isHadron"
    '''
    with open(filename, "rb") as f:
        particle_db = pickle.load(f)
    return particle_db


def set_default_qgsjetII_definition(particle_db):
    '''
    Also particles not explicitly known by QGSJetII may in fact interact via mapping 
    to cross section types (xsType) and hadron type (hadronType)

    This is achieved here.

    The function return nothing, but modified the input particle_db by adding the 
    fields 'xsType' and 'hadronType'
    '''
    for identifier, pData in particle_db.items():
        # the cross-section types
        xsType = "CannotInteract"
        hadronType = "UndefinedType"
        if (pData['isNucleus']):
            xsType = "Baryons"
            hadronType = "NucleusType"
        elif (pData['isHadron']):
            pdg = abs(pData['pdg'])
            anti = pData['pdg'] < 0
            isBaryon = (1000 <= pdg < 4000)
            charge = pData['charge']
            if (pdg>=100 and pdg<300 and pdg!=130): # light mesons
                xsType = "LightMesons"
                if (charge==0):
                    hadronType = "NeutralLightMesonType"
                else:
                    if (charge>0):
                        hadronType = "PiPlusType"
                    else:
                        hadronType = "PiMinusType"               
            elif ((pdg>=300 and pdg<400) or pdg in [130, 10313, 10323]): # kaons
                xsType = "Kaons"
                if (charge>0):
                    hadronType = "KaonPlusType"
                else:
                    hadronType = "KaonMinusType"
                if (charge==0):
                    hadronType = "Kaon0SType"
                    if (pdg == 130):
                        hadronType = "Kaon0LType"
                    elif (pdg == 310):
                        hadronType = "Kaon0SType"
            elif (isBaryon or pData['isNucleus']): # baryons
                xsType = "Baryons"
                if (charge==0):
                    if (anti):
                        hadronType = "AntiNeutronType"
                    else: 
                        hadronType = "NeutronType"
                else:
                    if (charge>0):
                        hadronType = "ProtonType"
                    else:
                        hadronType = "AntiProtonType"
            # all othe not-captured cased are hopefully irrelevant
            
        pData['qgsjetII_xsType'] = xsType
        pData['qgsjetII_hadronType'] = hadronType

            
def read_qgsjetII_codes(filename, particle_db):
    '''
    reads the qgsjet-codes data file. For particles known to QGSJetII the 'qgsjetII_code' is set in the particle_db, as
    well as the 'xsType' is updated in case it is different from its default value set above. 
    '''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line[0] == '#':
                continue
            line = line.split('#')[0]
            print ('QGSJetII codes: ', line)
            identifier, model_code, xsType = line.split()
            try:
                particle_db[identifier]["qgsjetII_code"] = int(model_code)
                particle_db[identifier]["qgsjetII_xsType"] = xsType
            except KeyError as e:
                raise Exception("Identifier '{:s}' not found in particle_db".format(identifier))

            
def generate_qgsjetII_enum(particle_db):
    '''
    generates the enum to access qgsjetII particles by readable names
    '''
    output = "enum class QgsjetIICode : int8_t {\n"
    for identifier, pData in particle_db.items():
        if 'qgsjetII_code' in pData:
            output += "  {:s} = {:d},\n".format(identifier, pData['qgsjetII_code'])
    output += "};\n"
    return output


def generate_corsika2qgsjetII(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetII codes
    '''
    string = "std::array<QgsjetIICode, {:d}> constexpr corsika2qgsjetII = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        if 'qgsjetII_code' in pData:
            string += "  QgsjetIICode::{:s}, \n".format(identifier)
        else:
            string += "  QgsjetIICode::Unknown, // {:s}\n".format(identifier + ' not implemented in QGSJetII')
    string += "};\n"
    return string
    

def generate_corsika2qgsjetII_xsType(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetII codes
    '''
    string = "std::array<QgsjetIIXSClass, {:d}> constexpr corsika2qgsjetIIXStype = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        modelCodeXS = pData.get("qgsjetII_xsType", "CannotInteract")
        string += "  QgsjetIIXSClass::{:s}, // {:s}\n".format(modelCodeXS, identifier if modelCodeXS else identifier + " (not implemented in QGSJETII)")
    string += "};\n"
    return string


def generate_corsika2qgsjetII_hadronType(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetII codes
    '''
    string = "std::array<QgsjetIIHadronType, {:d}> constexpr corsika2qgsjetIIHadronType = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        modelCode = pData.get("qgsjetII_hadronType", "UndefinedType")
        string += "  QgsjetIIHadronType::{:s}, // {:s}\n".format(modelCode, identifier if modelCode else identifier + " (not implemented in QGSJETII)")
    string += "};\n"
    return string


def generate_qgsjetII2corsika(particle_db) :
    '''
    generates the look-up table to convert qgsjetII codes to corsika codes    
    '''
    minID = 0
    for identifier, pData in particle_db.items() :
        if 'qgsjetII_code' in pData:
            minID = min(minID, pData['qgsjetII_code'])

    string = "QgsjetIICodeIntType constexpr minQgsjetII = {:d};\n\n".format(minID)

    pDict = {}
    for identifier, pData in particle_db.items() :
        if 'qgsjetII_code' in pData:
            model_code = pData['qgsjetII_code'] - minID
            pDict[model_code] = identifier
    
    nPart = max(pDict.keys()) - min(pDict.keys()) + 1
    string += "std::array<corsika::Code, {:d}> constexpr qgsjetII2corsika = {{\n".format(nPart)
    
    for iPart in range(nPart) :
        identifier = pDict.get(iPart, "Unknown")
        qgsID = iPart + minID
        string += "  corsika::Code::{:s}, // {:d} \n".format(identifier, qgsID)
    
    string += "};\n"
    return string

def generate_qgsjetII_start():
    string = "// This file is auto-generated. Do not edit!\n"
    string += "#pragma once\n"
    string += "namespace corsika::qgsjetII {\n"
    return string

def generate_qgsjetII_end():
    string = "}\n"
    return string



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: {:s} <particle_db.pkl> <qgsjetII_codes.dat>".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
        
    print("code_generator.py for QGSJETII")
    
    particle_db = load_particledb(sys.argv[1])
    read_qgsjetII_codes(sys.argv[2], particle_db)
    set_default_qgsjetII_definition(particle_db)

    with open("Generated.inc", "w") as f:
        print("// this file is automatically generated\n// edit at your own risk!\n", file=f)
        print(generate_qgsjetII_start(), file=f)
        print(generate_qgsjetII_enum(particle_db), file=f)
        print(generate_corsika2qgsjetII(particle_db), file=f)
        print(generate_qgsjetII2corsika(particle_db), file=f)
        print(generate_corsika2qgsjetII_xsType(particle_db), file=f)
        print(generate_corsika2qgsjetII_hadronType(particle_db), file=f)
        print(generate_qgsjetII_end(), file=f)
