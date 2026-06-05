#!/usr/bin/env bash


#alpha=0.0001 # diffmf_dev default
alpha=0.000001 # happy medium
#alpha=0.0000000001 # parallel_mf default 
max_deriv=2
fg_num_sigma=3

local_paths=0;
if [ "$1" == '--local' ]; then
    local_paths=1; shift;
fi

clobber=0;
if [ "$1" == '--clobber' ]; then
    clobber=1; shift;
fi

if [ ! "$#" == 1 ]; then
    echo "Usage: compare_cmf_diffmf.sh [--local] [--clobber] fid"
    return
fi

# below: default parallel_mf.py wvlrange for ch4
wvlrange="500 1340 1500 1790 1950 2450"
if [ "$1" == '--oldwvl' ]; then
    wvlrange="2137 2493"; shift;
elif [ "$1" == '--ch4wvl' ]; then
    wvlrange="1500 1790 1950 2450"; shift;
elif [ "$1" == '--allwvl' ]; then
    wvlrange="200 2600"; shift;
fi

fid=$1;
dstr=$(echo $fid|cut -c5-12)
if [ "$local_paths" == "1" ]; then
    ch4root=./mf_input
    rdnroot=./mf_input
else
    ch4root=/store/brodrick/methane/methane_20230813
    rdnroot=/store/emit/ops/data/acquisitions
fi

# input paths to radiance image, masks, target spectrum and instrument noise 
tgtpath=$ch4root/${dstr}/${fid}_ch4_target
rdnpath=$(ls -tr $rdnroot/${dstr}/${fid}/l1b/${fid}_o*_s*_l1b_rdn_b*_v*.img|head -n 1)
l1bmskf=$(ls -tr $rdnroot/${dstr}/${fid}/l1b/${fid}_o*_s*_l1b_bandmask_b*_v*.img|head -n 1)
l2amskf=$(ls -tr $rdnroot/${dstr}/${fid}/l2a/${fid}_o*_s*_l2a_mask_b*_v*.img|head -n 1)
npfpath=./instrument_noise_parameters/emit_noise.txt

# cmf output paths
outroot=./parallel_mf_output
outpath=${outroot}/${fid}_ch4
logpath=${outpath}_log.txt
uncpath=${outpath}_uncertainty
snspath=${outpath}_sensitivity

# diffmf output paths
doutroot=./parallel_diffmf_output
doutpath=${doutroot}/${fid}_ch4
dlogpath=${doutpath}_log.txt
duncpath=${doutpath}_uncertainty
dsnspath=${doutpath}_sensitivity
dfgmpath=${doutpath}_fgmsk

drefpath=${doutroot}/${fid}_ch4_refined
dreflogpath=${drefpath}_log.txt
drefuncpath=${drefpath}_uncertainty
drefsnspath=${drefpath}_sensitivity
dreffgmpath=${drefpath}_fgmsk

if [ ! -d ${outroot} ]; then
    mkdir -p $outroot
fi

if [ ! -d ${doutroot} ]; then
    mkdir -p $doutroot
fi
    
if [ "$clobber" == "1" ]; then
    if [ -f ${outpath} ]; then 
	rm -i ${outpath}{,.hdr}
    fi
    if [ -f ${doutpath} ]; then 
	rm -i ${doutpath}{,.hdr}
    fi
fi


echo "Processing $fid"
echo "rdnpath=$rdnpath"
echo "tgtpath=$tgtpath"

if [ ! -f $outpath ]; then 
    echo "Running parallel_mf.py with wavelength_range=$wvlrange"
    runcmd="python parallel_mf_orig.py $rdnpath $tgtpath $outpath \
	   --n_mc 1 --wavelength_range $wvlrange --fixed_alpha $alpha \
	   --l1b_bandmask_file $l1bmskf --l2a_mask_file $l2amskf \
	   --mask_clouds_water --mask_saturation --uncert_output_file $uncpath \
	   --sens_output_file $snspath --noise_parameters_file $npfpath \
	   --logfile $logpath"
    time $runcmd 
    echo "Done with base cmf retrieval."
else
    echo "Using existing cmf output: $outpath"
fi

if [ ! -f $doutpath ]; then 
    echo "Running parallel_diffmf.py with wavelength_range=$wvlrange max_deriv=$max_deriv"
    runcmd="python parallel_diffmf.py $rdnpath $tgtpath $doutpath --max_deriv $max_deriv \
       --n_mc 1 --wavelength_range $wvlrange --fixed_alpha $alpha \
       --l1b_bandmask_file $l1bmskf --l2a_mask_file $l2amskf \
       --mask_clouds_water --mask_saturation --noise_parameters_file $npfpath \
       --sens_output_file $dsnspath --uncert_output_file $duncpath \
       --logfile $dlogpath --fg_num_sigma $fg_num_sigma \
       --fg_output_file $dfgmpath"
    time $runcmd
    echo "Done with base diffmf retrieval."
fi

if [ ! -f $drefpath ]; then 
    echo "Refining parallel_diffmf.py output with wavelength_range=$wvlrange max_deriv=$max_deriv"
    runcmd="python parallel_diffmf.py $rdnpath $tgtpath $drefpath --max_deriv $max_deriv \
       --n_mc 1 --wavelength_range $wvlrange --fixed_alpha $alpha \
       --l1b_bandmask_file $l1bmskf --l2a_mask_file $l2amskf \
       --mask_clouds_water --mask_saturation --noise_parameters_file $npfpath \
       --sens_output_file $drefsnspath --uncert_output_file $drefuncpath \
       --logfile $dreflogpath --fg_num_sigma $fg_num_sigma \
        --fg_output_file $dreffgmpath --fg_input_file $dfgmpath"
       
    time $runcmd
    echo "Done with refined diffmf retrieval."    
else
    echo "Using existing diffmf output: $drefpath"
fi

# Output CMF (single-channel) vs DiffMF (${max_deriv}-channel) stats
# CMF outputs should be within \eps of DiffMF for first channel only

rm ${outroot}/*.aux.xml
rm ${doutroot}/*.aux.xml

echo $outpath
gdalinfo -stats $outpath|grep -A1 "Band 1"
echo $doutpath
gs=$(gdalinfo -stats $doutpath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done
echo $drefpath
gs=$(gdalinfo -stats $drefpath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done
echo

# CMF vs. DiffMF uncertainty
echo $uncpath
gdalinfo -stats $uncpath|grep -A1 "Band 1"
echo $duncpath
gs=$(gdalinfo -stats $duncpath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done
echo $drefuncpath
gs=$(gdalinfo -stats $drefuncpath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done
echo

# CMF vs. DiffMF sensitivity
echo $snspath
gdalinfo -stats $snspath|grep -A1 "Band 1"
echo $dsnspath
gs=$(gdalinfo -stats $dsnspath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done
echo $drefsnspath
gs=$(gdalinfo -stats $drefsnspath)
for i in $(seq 3); do echo "$gs"|grep -A1 "Band $i"; done


