import json
import subprocess
import typing
from collections import defaultdict
from pathlib import Path
import os
import Bio.PDB
import numpy as np
import prody as pd
import pandas
import requests
from Bio import SeqIO

path_type = typing.Union[str, Path]

to_one_letter_code = {'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'ILE': 'I', 'LEU': 'L', 'ASP': 'D',
                      'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H', 'CYS': 'C',
                      'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G', '2AS': 'D', '3AH': 'H', '5HP': 'E', 'ACL': 'R',
                      'AIB': 'A', 'ALM': 'A', 'ALO': 'T', 'ALY': 'K', 'ARM': 'R', 'ASA': 'D', 'ASB': 'D', 'ASK': 'D',
                      'ASL': 'D', 'ASQ': 'D', 'AYA': 'A', 'BCS': 'C', 'BHD': 'D', 'BMT': 'T', 'BNN': 'A', 'BUC': 'C',
                      'BUG': 'L', 'C5C': 'C', 'C6C': 'C', 'CCS': 'C', 'CEA': 'C', 'CHG': 'A', 'CLE': 'L', 'CME': 'C',
                      'CSD': 'A', 'CSO': 'C', 'CSP': 'C', 'CSS': 'C', 'CSW': 'C', 'CXM': 'M', 'CY1': 'C', 'CY3': 'C',
                      'CYG': 'C', 'CYM': 'C', 'CYQ': 'C', 'DAH': 'F', 'DAL': 'A', 'DAR': 'R', 'DAS': 'D', 'DCY': 'C',
                      'DGL': 'E', 'DGN': 'Q', 'DHA': 'A', 'DHI': 'H', 'DIL': 'I', 'DIV': 'V', 'DLE': 'L', 'DLY': 'K',
                      'DNP': 'A', 'DPN': 'F', 'DPR': 'P', 'DSN': 'S', 'DSP': 'D', 'DTH': 'T', 'DTR': 'W', 'DTY': 'Y',
                      'DVA': 'V', 'EFC': 'C', 'FLA': 'A', 'FME': 'M', 'GGL': 'E', 'GLZ': 'G', 'GMA': 'E', 'GSC': 'G',
                      'HAC': 'A', 'HAR': 'R', 'HIC': 'H', 'HIP': 'H', 'HMR': 'R', 'HPQ': 'F', 'HTR': 'W', 'HYP': 'P',
                      'IIL': 'I', 'IYR': 'Y', 'KCX': 'K', 'LLP': 'K', 'LLY': 'K', 'LTR': 'W', 'LYM': 'K', 'LYZ': 'K',
                      'MAA': 'A', 'MEN': 'N', 'MHS': 'H', 'MIS': 'S', 'MLE': 'L', 'MPQ': 'G', 'MSA': 'G', 'MSE': 'M',
                      'MVA': 'V', 'NEM': 'H', 'NEP': 'H', 'NLE': 'L', 'NLN': 'L', 'NLP': 'L', 'NMC': 'G', 'OAS': 'S',
                      'OCS': 'C', 'OMT': 'M', 'PAQ': 'Y', 'PCA': 'E', 'PEC': 'C', 'PHI': 'F', 'PHL': 'F', 'PR3': 'C',
                      'PRR': 'A', 'PTR': 'Y', 'SAC': 'S', 'SAR': 'G', 'SCH': 'C', 'SCS': 'C', 'SCY': 'C', 'SEL': 'S',
                      'SEP': 'S', 'SET': 'S', 'SHC': 'C', 'SHR': 'K', 'SOC': 'C', 'STY': 'Y', 'SVA': 'S', 'TIH': 'A',
                      'TPL': 'W', 'TPO': 'T', 'TPQ': 'A', 'TRG': 'K', 'TRO': 'W', 'TYB': 'Y', 'TYQ': 'Y', 'TYS': 'Y',
                      'TYY': 'Y', 'AGM': 'R', 'GL3': 'G', 'SMC': 'C', 'ASX': 'B', 'CGU': 'E', 'CSX': 'C', 'GLX': 'Z',
                      'PYX': 'C', 'UNK': 'X'}


def load_database(json_file) -> dict:
    """
    Loads characterized STS database from a JSON file

    Parameters
    ----------
    json_file

    Returns
    -------
    dict of UniProt ID to dict containing entry information
    """
    json_data = json.load(open(json_file))["data"]
    dict_data = {}
    for entry in json_data:
        if not len(entry["Major Product Cyclization"].strip()):
            entry["Major Product Cyclization"] = "acyclic"
        dict_data[entry["UniProt ID"]] = entry
    return dict_data


def get_uniprot_sequence(uniprot_id: str) -> str:
    """
    Get sequence of uniprot_id
    Parameters
    ----------
    uniprot_id

    Returns
    -------
    sequence
    None if not found
    """
    mapping_url = f"http://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(mapping_url).text
    if "html" not in response.strip():
        return ''.join(response.split('\n')[1:]).upper()
    else:
        mapping_url = f"http://www.uniprot.org/uniparc/{uniprot_id}.fasta"
        response = requests.get(mapping_url).text
        if "html" not in response.strip():
            return ''.join(response.split('\n')[1:]).upper()
        else:
            return None


def get_file_parts(input_filename: path_type) -> tuple:
    """
    Gets directory path, name, and extension from a filename
    Parameters
    ----------
    input_filename

    Returns
    -------
    (path, name, extension)
    """
    input_filename = Path(input_filename)
    path = str(input_filename.parent)
    extension = input_filename.suffix
    name = input_filename.stem
    return path, name, extension


def read_pdb(input_file: path_type, name: str = None, chain: str = None) -> tuple:
    """
    returns protein information from PDB file
    Parameters
    ----------
    input_file
    name
        None => takes from filename
    chain
        only for that chain

    Returns
    -------
    structure Object, list of residue Objects, list of peptide Objects, sequence, sequence to residue index
    """
    if name is None:
        (_, name, _) = get_file_parts(input_file)
    input_file = str(input_file)
    structure = Bio.PDB.PDBParser().get_structure(name, input_file)
    if chain is not None:
        structure = structure[0][chain]
    residues = Bio.PDB.Selection.unfold_entities(structure, 'R')
    peptides = Bio.PDB.PPBuilder().build_peptides(structure)
    sequence = ''.join([str(peptide.get_sequence()) for peptide in peptides])
    residue_dict = dict(zip(residues, range(len(residues))))
    seq_to_res_index = [residue_dict[r] for peptide in peptides for r in peptide]
    return structure, residues, peptides, sequence, seq_to_res_index


def load_protein_prody(pdb_file, key=None, chain=None):
    """
    Loads a pdb_file into a ProDy protein object

    Parameters
    ----------
    pdb_file
    key
    chain

    Returns
    -------
    ProDy AtomGroup
    """
    pdb_file = str(pdb_file)
    if key is None:
        _, key, _ = get_file_parts(pdb_file)
    if chain is None:
        if get_file_parts(pdb_file)[-1] == ".pqr":
            protein = pd.parsePQR(pdb_file, title=key)
        else:
            protein = pd.parsePDB(pdb_file, title=key)
    else:
        if get_file_parts(pdb_file)[-1] == ".pqr":
            protein = pd.parsePQR(pdb_file, title=key, chain=chain)
        else:
            protein = pd.parsePDB(pdb_file, title=key, chain=chain)
    return protein


def get_sequences_from_pdb_files(pdb_files: list, output_sequence_file: path_type):
    """
    Extract PDB sequences from a list of PDB files into a fasta file

    Parameters
    ----------
    pdb_files
    output_sequence_file
    """
    with open(output_sequence_file, "w") as f:
        for pdb_file in pdb_files:
            pdb_file = str(pdb_file)
            _, name, _ = get_file_parts(pdb_file)
            _, _, _, sequence, _ = read_pdb(pdb_file)
            f.write(f">{name}\n{sequence.upper()}\n")


def get_hmm_sequences(sequence_file: path_type, hmm_file: path_type, subsequence_file: path_type):
    """
    retrieve portions of sequences matching an HMM.

    Parameters
    ----------
    sequence_file
    hmm_file
    subsequence_file

    Returns
    -------
    subsequence file
    """
    output_path, output_name, _ = get_file_parts(subsequence_file)
    hmm_sequences_aln_file = Path(output_path) / f"{output_name}.sto"
    hmmsearch(sequence_file, hmm_sequences_aln_file, hmm_file)
    if hmm_sequences_aln_file.stat().st_size > 0:
        reformat_to_fasta(hmm_sequences_aln_file, subsequence_file)
        retain_largest_domain(subsequence_file)
        Path(hmm_sequences_aln_file).unlink()
        return subsequence_file


def retain_largest_domain(fasta_file: path_type):
    """
    Change a subsequence file such that each ID only has one subsequence (the largest)

    Parameters
    ----------
    fasta_file

    """

    def prune_header(key):
        if "/" in key:
            key = key.split("/")[0].strip()
        if "|" in key:
            key = key.split("|")[1].strip()
        return key

    sequences = get_sequences_from_fasta(fasta_file, prune_headers=False)
    headers = defaultdict(list)
    for key in sequences.keys():
        headers[prune_header(key)].append(key)
    with open(fasta_file, "w") as f:
        for key in headers:
            header = sorted([(len(sequences[h]), h) for h in headers[key]], reverse=True)[0][1]
            f.write(">" + header + "\n")
            f.write(sequences[header].strip() + "\n")


def hmmsearch(input_sequence_file: path_type, output_alignment_file: path_type, input_hmm_file: path_type, use_max=True, ince=None):
    """
    Runs hmmsearch
    Parameters
    ----------
    output_alignment_file
    input_hmm_file
    input_sequence_file
    use_max
    ince
    """
    output_alignment_file = str(output_alignment_file)
    input_hmm_file = str(input_hmm_file)
    input_sequence_file = str(input_sequence_file)
    if use_max:
        if ince is not None:
            subprocess.check_call(
                f"hmmsearch --noali --max --incE {ince} --incdomE {ince} -o log_hmm.out -A {output_alignment_file} {input_hmm_file} {input_sequence_file}",
                shell=True)
        else:
            subprocess.check_call(f"hmmsearch --noali --max -o log_hmm.out -A {output_alignment_file} {input_hmm_file} {input_sequence_file}",
                                  shell=True)
    else:
        subprocess.check_call(f"hmmsearch --noali -o log_hmm.out -A {output_alignment_file} {input_hmm_file} {input_sequence_file}",
                              shell=True)


def hmmscore(output_table_file, input_hmm_file, input_sequence_file):
    path, name, _ = get_file_parts(output_table_file)
    aln_file = Path(path) / f"{name}.aln"
    sto_file = Path(path) / f"{name}.sto"
    subprocess.check_call(f"hmmsearch --nobias -o {aln_file} -A {sto_file} --tblout {output_table_file} {input_hmm_file} {input_sequence_file}",
                          shell=True)


def get_score_from_hmmsearch_table(table_file: str, prune_headers=True, score=False) -> dict:
    table = np.genfromtxt(table_file, dtype='str', comments="#")
    if table.size == 0:
        return {}
    table = np.atleast_2d(table)
    if score:
        table = table[:, [0, 2, 5]]
    else:
        table = table[:, [0, 2, 4]]
    scores = {}
    if prune_headers:
        for row in table:
            if row[0].split("/")[0] not in scores:
                scores[row[0].split("/")[0]] = {}
            scores[row[0].split("/")[0]][row[1].split("_")[1]] = float(row[2])
    else:
        for row in table:
            if row[0] not in scores:
                scores[row[0]] = {}
            scores[row[0]][row[1].split("_")[1]] = float(row[2])
    return scores


def reformat_to_fasta(input_file: path_type, output_file: path_type):
    """
    Changes sto file to fasta file

    Parameters
    ----------
    input_file
    output_file
    """
    subprocess.check_call(f"esl-reformat fasta {str(input_file)} > {str(output_file)}", shell=True)


def get_sequences_from_fasta_yield(fasta_file: typing.Union[str, Path], prune_headers: bool = True) -> tuple:
    """
    Returns (accession, sequence) iterator
    Parameters
    ----------
    fasta_file
    prune_headers
        only keeps accession upto first /

    Returns
    -------
    (accession, sequence)
    """
    with open(fasta_file) as f:
        current_sequence = ""
        current_key = None
        for line in f:
            if not len(line.strip()):
                continue
            if "==" in line:
                continue
            if ">" in line:
                if current_key is None:
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
                    if "|" in current_key and prune_headers:
                        current_key = current_key.split("|")[1].strip()
                else:
                    yield (current_key, current_sequence)
                    current_sequence = ""
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
                    if "|" in current_key and prune_headers:
                        current_key = current_key.split("|")[1].strip()
            else:
                current_sequence += line.strip()
        yield (current_key, current_sequence)


def get_sequences_from_fasta(fasta_file: typing.Union[str, Path], prune_headers: bool = True) -> dict:
    """
    Returns dict of accession to sequence from fasta file
    Parameters
    ----------
    fasta_file
    prune_headers
        only keeps accession upto first /

    Returns
    -------
    {accession:sequence}
    """
    return {key: sequence for (key, sequence) in get_sequences_from_fasta_yield(fasta_file, prune_headers)}


def get_start_end_residues(pdb_files: list, subsequence_file: path_type) -> tuple:
    """
    Finds the start (residue position & chain) and end of subsequences in a list of PDB files
    Parameters
    ----------
    pdb_files
    subsequence_file
        returned by hmmer + esl-reformat (contains start-end of subsequence in fasta header)

    Return
    -------
    dict of {PDB_ID: (start position, start chain, end position, end chain)}
    dict of {PDB_ID: (sequence_start, sequence_end)
    """
    sequence_start_end = {}
    subsequences = get_sequences_from_fasta(subsequence_file, prune_headers=False)
    for key in subsequences:
        if len(key.split('/')) == 1:
            name = key
            start, end = 1, len(subsequences[key])
        else:
            name, start_end = key.split("/")
            start, end = start_end.split()[0].split("-")
        sequence_start_end[name] = (int(start) - 1, int(end) - 1)
    pdb_start_end = {}
    for pdb_file in pdb_files:
        pdb_file = str(pdb_file)
        _, name, _ = get_file_parts(pdb_file)
        _, residues, _, sequence, seq_to_res_index = read_pdb(pdb_file)
        seq_start, seq_end = sequence_start_end[name]
        (_, _, start_chain, (_, res_start, _)) = residues[seq_to_res_index[seq_start]].get_full_id()
        (_, _, end_chain, (_, res_end, _)) = residues[seq_to_res_index[seq_end]].get_full_id()
        assert to_one_letter_code[residues[seq_to_res_index[seq_start]].get_resname()] == sequence[seq_start]
        assert to_one_letter_code[residues[seq_to_res_index[seq_end]].get_resname()] == sequence[seq_end]
        pdb_start_end[name] = (start_chain, res_start, end_chain, res_end)
    return pdb_start_end, sequence_start_end


def clustal_msa_from_sequences(input_sequence_file: path_type, output_alignment_file: path_type, input_hmm_file: path_type = None,
                               distance_matrix_file: path_type = None, n_iter=2, full=False, threads=30, log_dir=".", run=True):
    """
    Run clustalo

    Parameters
    ----------
    input_sequence_file
    output_alignment_file
    input_hmm_file
        if not None, hmm is used to align
    distance_matrix_file
        if not None, distance matrix is calculated
    n_iter
    full
    threads
    log_dir
    """
    input_sequence_file = str(input_sequence_file)
    output_alignment_file = str(output_alignment_file)
    if input_hmm_file is not None:
        input_hmm_file = str(input_hmm_file)
        if distance_matrix_file is None:
            if full:
                command = f"clustalo --iter={n_iter} --full-iter --output-order=input-order --log={log_dir}/log_clustal.out -i {input_sequence_file} --hmm-in={input_hmm_file} -o {output_alignment_file} --threads={threads} --force -v"
            else:
                command = f"clustalo --iter={n_iter} --output-order=input-order --log={log_dir}/log_clustal.out -i {input_sequence_file} --hmm-in={input_hmm_file} -o {output_alignment_file} --threads={threads} --force -v"

        else:
            distance_matrix_file = str(distance_matrix_file)
            command = f"clustalo --iter={n_iter} --output-order=input-order --log={log_dir}/log_clustal.out --full --distmat-out={distance_matrix_file} -i {input_sequence_file} --hmm-in={input_hmm_file} -o {output_alignment_file} --threads={threads} --force -v"
    else:
        if distance_matrix_file is None:
            if full:
                command = f"clustalo --iter={n_iter} --full-iter --output-order=input-order --log={log_dir}/log_clustal.out -i {input_sequence_file} -o {output_alignment_file} --threads={threads} --force -v"
            else:
                command = f"clustalo --iter={n_iter} --output-order=input-order --log={log_dir}/log_clustal.out -i {input_sequence_file} -o {output_alignment_file} --threads={threads} --force -v"
        else:
            distance_matrix_file = str(distance_matrix_file)
            command = f"clustalo --iter={n_iter} --output-order=input-order --log={log_dir}/log_clustal.out --full --distmat-out={distance_matrix_file} -i {input_sequence_file} -o {output_alignment_file} --threads={threads} --force -v"
    if run:
        return subprocess.check_call(command, shell=True)
    else:
        return command


def clustal_msa_from_sequences_and_profile(input_sequence_file, profile_sequence_file, output_file, n_iter=2, full=False,
                                           threads=30, log_dir=".", run=True):
    """
    Run clustalo with an aligned profile and a new set of sequences
    Parameters
    ----------
    input_sequence_file
    profile_sequence_file
    output_file
    n_iter
    full
    threads
    log_dir
    run

    Returns
    -------
    if run=True then subprocess, else the command to run
    """
    if full:
        command = f"clustalo --iter={n_iter} --full-iter --log={log_dir}/log_clustal.out -i {input_sequence_file} --p1={profile_sequence_file} -o {output_file} --threads={threads} --force -v"
    else:
        command = f"clustalo --iter={n_iter} --log={log_dir}/log_clustal.out -i {input_sequence_file} --p1={profile_sequence_file} -o {output_file} --threads={threads} --force -v"
    if run:
        return subprocess.check_call(
            command,
            shell=True)
    else:
        return command


def mkdir(directory):
    """
    makes a new directory if it doesn't exist
    Parameters
    ----------
    directory

    Returns
    -------
    directory
    """
    directory = Path(directory)
    if not directory.is_dir():
        directory.mkdir()
    return directory


def group_indices(input_list: list) -> list:
    """
    [1, 1, 1, 2, 2, 3, 3, 3, 4] -> [[0, 1, 2], [3, 4], [5, 6, 7], [8]]
    Parameters
    ----------
    input_list

    Returns
    -------
    list of lists
    """
    output_list = []
    current_list = []
    current_index = None
    for i in range(len(input_list)):
        if current_index is None:
            current_index = input_list[i]
        if input_list[i] == current_index:
            current_list.append(i)
        else:
            output_list.append(current_list)
            current_list = [i]
        current_index = input_list[i]
    output_list.append(current_list)
    return output_list

def make_ali_file(key, key_sequence_file, template_file, output_dir, env):
    """
    using clustalo to make alignment
    :param key: the sequence ID list
    :param key_sequence_file: the sequence fasta file of all keys
    :param template_file:the template fasta file
    :param output_dir:
    :param env:
    :return:
    """
    f = open(f"{output_dir}/{key}.fasta", 'w')
    for seq_record in SeqIO.parse(template_file, "fasta"):
        f.write(f">{seq_record.id}\n")
        f.write(f"{seq_record.seq}\n")
    for seq_record in SeqIO.parse(key_sequence_file, "fasta"):
        if seq_record.id == key:
            f.write(f">{seq_record.id}\n")
            f.write(f"{seq_record.seq}\n")
    f.close()
    command = f"clustalo  -i {output_dir}/{key}.fasta  -o {output_dir}/{key}_aln.fasta"
    subprocess.check_call(command, shell=True, env=env)
    os.remove(f"./{output_dir}/{key}.fasta")
    return get_sequences_from_fasta(f"{output_dir}/{key}_aln.fasta")

def make_ali_file_2(keys_sequence, output_dir, env):
    """
    using clustalo to make alignment
    :param keys_sequence: the dictionary of all sequence ID and sequences
    :param output_dir:
    :param env:
    :return:
    """
    f = open(f"{output_dir}/unsuperviesed_learning.fasta", 'w')
    for key in keys_sequence.keys():
        f.write(f">{key}\n")
        f.write(f"{keys_sequence[key]}\n")
    f.close()
    command = f"clustalo  -i {output_dir}/unsuperviesed_learning.fasta  -o {output_dir}/unsuperviesed_learning_aln.fasta --force"
    subprocess.check_call(command, shell=True, env=env)
    os.remove(f"./{output_dir}/unsuperviesed_learning.fasta")
    return get_sequences_from_fasta(f"{output_dir}/unsuperviesed_learning_aln.fasta")

