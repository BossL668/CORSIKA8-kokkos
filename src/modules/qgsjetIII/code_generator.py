#!/usr/bin/env python3

# (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
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


def set_default_qgsjetIII_definition(particle_db):
    '''
    Also particles not explicitly known by QGSJetIII may in fact interact via mapping 
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
            
        pData['qgsjetIII_xsType'] = xsType
        pData['qgsjetIII_hadronType'] = hadronType

            
def read_qgsjetIII_codes(filename, particle_db):
    '''
    reads the qgsjet-codes data file. For particles known to QGSJetIII the 'qgsjetIII_code' is set in the particle_db, as
    well as the 'xsType' is updated in case it is different from its default value set above. 
    '''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line[0] == '#':
                continue
            line = line.split('#')[0]
            print ('QGSJetIII codes: ', line)
            identifier, model_code, xsType = line.split()
            try:
                particle_db[identifier]["qgsjetIII_code"] = int(model_code)
                particle_db[identifier]["qgsjetIII_xsType"] = xsType
            except KeyError as e:
                raise Exception("Identifier '{:s}' not found in particle_db".format(identifier))

            
def generate_qgsjetIII_enum(particle_db):
    '''
    generates the enum to access qgsjetIII particles by readable names
    '''
    output = "enum class QgsjetIIICode : int8_t {\n"
    for identifier, pData in particle_db.items():
        if 'qgsjetIII_code' in pData:
            output += "  {:s} = {:d},\n".format(identifier, pData['qgsjetIII_code'])
    output += "};\n"
    return output


def generate_corsika2qgsjetIII(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetIII codes
    '''
    string = "std::array<QgsjetIIICode, {:d}> constexpr corsika2qgsjetIII = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        if 'qgsjetIII_code' in pData:
            string += "  QgsjetIIICode::{:s}, \n".format(identifier)
        else:
            string += "  QgsjetIIICode::Unknown, // {:s}\n".format(identifier + ' not implemented in QGSJetIII')
    string += "};\n"
    return string
    

def generate_corsika2qgsjetIII_xsType(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetIII codes
    '''
    string = "std::array<QgsjetIIIXSClass, {:d}> constexpr corsika2qgsjetIIIXStype = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        modelCodeXS = pData.get("qgsjetIII_xsType", "CannotInteract")
        string += "  QgsjetIIIXSClass::{:s}, // {:s}\n".format(modelCodeXS, identifier if modelCodeXS else identifier + " (not implemented in QGSJETIII)")
    string += "};\n"
    return string


def generate_corsika2qgsjetIII_hadronType(particle_db):    
    '''
    generates the look-up table to convert corsika codes to qgsjetIII codes
    '''
    string = "std::array<QgsjetIIIHadronType, {:d}> constexpr corsika2qgsjetIIIHadronType = {{\n".format(len(particle_db))
    for identifier, pData in particle_db.items():
        if pData['isNucleus']: continue
        modelCode = pData.get("qgsjetIII_hadronType", "UndefinedType")
        string += "  QgsjetIIIHadronType::{:s}, // {:s}\n".format(modelCode, identifier if modelCode else identifier + " (not implemented in QGSJETIII)")
    string += "};\n"
    return string


def generate_qgsjetIII2corsika(particle_db) :
    '''
    generates the look-up table to convert qgsjetIII codes to corsika codes    
    '''
    minID = 0
    for identifier, pData in particle_db.items() :
        if 'qgsjetIII_code' in pData:
            minID = min(minID, pData['qgsjetIII_code'])

    string = "QgsjetIIICodeIntType constexpr minQgsjetIII = {:d};\n\n".format(minID)

    pDict = {}
    for identifier, pData in particle_db.items() :
        if 'qgsjetIII_code' in pData:
            model_code = pData['qgsjetIII_code'] - minID
            pDict[model_code] = identifier
    
    nPart = max(pDict.keys()) - min(pDict.keys()) + 1
    string += "std::array<corsika::Code, {:d}> constexpr qgsjetIII2corsika = {{\n".format(nPart)
    
    for iPart in range(nPart) :
        identifier = pDict.get(iPart, "Unknown")
        qgsID = iPart + minID
        string += "  corsika::Code::{:s}, // {:d} \n".format(identifier, qgsID)
    
    string += "};\n"
    return string

def generate_qgsjetIII_start():
    string = "// This file is auto-generated. Do not edit!\n"
    string += "#pragma once\n"
    string += "namespace corsika::qgsjetIII {\n"
    return string

def generate_qgsjetIII_end():
    string = "}\n"
    return string



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: {:s} <particle_db.pkl> <qgsjetIII_codes.dat>".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
        
    print("code_generator.py for QGSJETIII")
    
    particle_db = load_particledb(sys.argv[1])
    read_qgsjetIII_codes(sys.argv[2], particle_db)
    set_default_qgsjetIII_definition(particle_db)

    with open("Generated.inc", "w") as f:
        print("// this file is automatically generated\n// edit at your own risk!\n", file=f)
        print(generate_qgsjetIII_start(), file=f)
        print(generate_qgsjetIII_enum(particle_db), file=f)
        print(generate_corsika2qgsjetIII(particle_db), file=f)
        print(generate_qgsjetIII2corsika(particle_db), file=f)
        print(generate_corsika2qgsjetIII_xsType(particle_db), file=f)
        print(generate_corsika2qgsjetIII_hadronType(particle_db), file=f)
        print(generate_qgsjetIII_end(), file=f)
