#!/bin/env perl
use strict;
require "../lib/libSystem.pl";
$|=1;

my($indir, $outfile) = @ARGV;
$indir ||= "data";
$outfile ||= "results/drug-indi";
my $drugsFDAdir = "../DrugsFDA";
my $drugsFDAproducts = "unzip -p $drugsFDAdir/ndctext.zip product.txt |";

my %ignore = (
	'product used for unknown indication', 1,
	'off label use', 1,
	'prophylaxis', 1,
	'ill-defined disorder', 1,
	'premedication', 1,
	'product use in unapproved indication', 1,
	'intentional product misuse', 1,
	'exposure during pregnancy', 1,
	'foetal exposure during pregnancy', 1,
);

# learn which cases should be ignored
my $delete = readDELETE($indir);

# get list of data quarters, from most recent to oldest
my @quarters = findQuarters($indir);

# read info on drug approvals
#my($NDAname, $NDAing, $NDAfor, $ingIn) = readNDAnames("$drugsFDAdir/data/latest/Products.txt");
my($NDAname, $NDAing, $NDAfor, $ingIn) = readNDAproducts($drugsFDAproducts);

my %seenCase;
my %drugName_indi;
my %drugNDA_indi;
my %nda;
my %ndadrugs;
my $done;
foreach my $q (@quarters) {
	print "#$q";
	my($case_drugName, $case_drugIng, $case_drugNDA, $seenInQ) = readDRUG("$indir/DRUG$q.txt.gz");
	print "\t", scalar keys %$case_drugNDA;
	print "\t", scalar keys %$case_drugName;
	readINDI("$indir/INDI$q.txt.gz", $case_drugName, $case_drugIng, $case_drugNDA, $seenInQ);
	print "\t", scalar keys %drugNDA_indi;
	print "\t", scalar keys %drugName_indi;
	print "\n";
	$done++;
	#last if $done>0;
}

foreach my $drug (sort keys %nda) {
	last; ######
	my %ing;
	next unless $nda{$drug};
	foreach my $nda (keys %{$nda{$drug}}) {
		foreach my $ing (keys %{$NDAing->{$nda}}) {
			$ing{$ing}++;
		}
	}
	print join("\t", $drug, join(",", keys %{$nda{$drug}}), join("##", keys %ing)), "\n";
}


open OUTF, "| sort -k1rn | gzip -c > $outfile-nda.txt.gz";
print OUTF join("\t", qw/CASES NDA DRUGNAME INGREDIENTS INDICATION/), "\n";
foreach my $nda (sort keys %drugNDA_indi) {
	foreach my $indication (sort keys %{$drugNDA_indi{$nda}}) {
		my $drugname = join("|", sort keys %{$NDAname->{$nda}});
		my $ingredients = join("|", sort keys %{$NDAing->{$nda}});
		print OUTF join("\t", $drugNDA_indi{$nda}{$indication}, $nda, $drugname, $ingredients, $indication), "\n";
	}
}
close OUTF;

open OUTF, ">$outfile-name.txt";
foreach my $drugname (sort keys %drugName_indi) {
		foreach my $indication (sort keys %{$drugName_indi{$drugname}}) {
		my $ingredients = join("|", sort keys %{$ingIn->{$drugname}});
				print OUTF join("\t", $drugName_indi{$drugname}{$indication}, $NDAfor->{$drugname} || 'NA', $drugname, $ingredients, $indication), "\n";
		}
}
close OUTF;


###
sub readNDAproducts {
	my($file) = @_;

	my(%name, %ing, %ndaFor, %ingIn);
	open F, $file;
	$_ = <F>;
	chomp;
	s/\r//g;
	my @headers = split /\t/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	# PRODUCTID	PRODUCTNDC	PRODUCTTYPENAME	PROPRIETARYNAME	PROPRIETARYNAMESUFFIX	NONPROPRIETARYNAME	DOSAGEFORMNAME	ROUTENAME	STARTMARKETINGDATE	ENDMARKETINGDATE	MARKETINGCATEGORYNAME	APPLICATIONNUMBER	LABELERNAME	SUBSTANCENAME	ACTIVE_NUMERATOR_STRENGTH	ACTIVE_INGRED_UNIT	PHARM_CLASSES	DEASCHEDULE	NDC_EXCLUDE_FLAG	LISTING_RECORD_CERTIFIED_THROUGH
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\t/;
		my $nda = $v[$col{'APPLICATIONNUMBER'}];
		next unless $nda;
		my $drugname = lc $v[$col{'PROPRIETARYNAME'}];
		my $ingredient = lc $v[$col{'NONPROPRIETARYNAME'}];
		$name{$nda}{$drugname}++;
		$ing{$nda}{$ingredient}++;
		$ndaFor{$drugname}{$nda}++;
		$ndaFor{$ingredient}{$nda}++;
		$ingIn{$drugname}{$ingredient}++;

		if ($nda =~ /^(NDA|BLA|ANDA)0*(.+)/) {
			$nda = $2;
			next unless $nda;
			$name{$nda}{$drugname}++;
			$ing{$nda}{$ingredient}++;
		}
	}
	close F;
	foreach my $term (keys %ndaFor) {
		$ndaFor{$term} = join("|", sort keys %{$ndaFor{$term}});
	}
	return \%name, \%ing, \%ndaFor, \%ingIn;
}
sub readNDAnames {
	my($file) = @_;
	my(%name, %ing, %ndaFor, %ingIn);
	open F, $file;
	$_ = <F>;
	while (<F>) {
		chomp;
		s/\r//g;
		my($nda, undef, undef, undef, undef, $drugname, $ingredient) = split /\t/;
		$drugname = lc $drugname;
		$ingredient = lc $ingredient;
		# refdrug can be null, 0, 1, 2
		#$ref{$nda}{$refdrug}++;
		$name{$nda}{$drugname}++;
		$ing{$nda}{$ingredient}++;
		#$ndaFor{$refdrug}{$nda}++;
		$ndaFor{$drugname}{$nda}++;
		$ndaFor{$ingredient}{$nda}++;
		$ingIn{$drugname}{$ingredient}++;
	}
	close F;
	foreach my $term (keys %ndaFor) {
		$ndaFor{$term} = join("|", sort keys %{$ndaFor{$term}});
		#print join("\t", $term, $ndaFor{$term}), "\n" if $ndaFor{$term} =~ /\|/;
	}
	return \%name, \%ing, \%ndaFor, \%ingIn;
}

sub readDELETE {
	my($dir) = @_;
	my %info;
	foreach my $file (slicedirlist($dir, "^DELETE")) {
		open F, "gunzip -c $dir/$file |";
		while (<F>) {
			chomp;
			s/\r//g;
			$info{$_}++;
		}
		close F;
	}
	return \%info;
}

sub findQuarters {
	my($dir) = @_;
	my %info;
	foreach my $file (fulldirlist($indir)) {
		my($q) = $file =~ /(\d\dQ\d)/i;
		$info{$q}++ if $q;
	}
	return reverse sort keys %info;
	#return sort keys %info;
}

sub readDRUG {
	my($file) = @_;
	my %drugName;
	my %drugIng;
	my %drugNDA;
	my %seenInQ;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		my $caseid = $v[$col{'caseid'}];
		if ($caseid) {
			next if $seenCase{$caseid};
			$seenCase{$caseid} = 1;
			$seenInQ{$caseid} = 1;
		} else {
			$seenInQ{$primary} = 1;
		}
		my $drugname = lc $v[$col{'drugname'}];
		$drugname =~ s/^,//;
		$drugname =~ s/^\s+//;
		$drugname =~ s/\s+$//;
		$drugname =~ s/\s\(\s*\)\s*$//;
		my $drugseq = $v[$col{'drug_seq'}] if defined $col{'drug_seq'};
		my $ingredient = lc $v[$col{'prod_ai'}] if defined $col{'prod_ai'};
		my $nda = $v[$col{'nda_num'}];
		$nda =~ s/^0+//;
		#print join("\t", $file, $primary, $drugseq || 'NA', $drugname, $ingredient || 'NA', $nda || 'NNDA'), "\n";
		if ($ingredient) {
			#print "new alias $ingredient for $drugname\n" unless $ingIn->{$drugname}{$ingredient};
			$ingIn->{$drugname}{$ingredient}++;
		}
		my $case = "$primary\$$drugseq";
		$drugName{$case} = $drugname;
		$drugIng{$case} = $ingredient if $ingredient;
		#if ($nda && !defined $NDAname->{$nda}) {
		#	print " $nda";
		#}
		$drugNDA{$case} = $nda if $nda && defined $NDAname->{$nda};
		
		$nda{$drugname}{$nda}++ if $nda && defined $NDAname->{$nda};# $nda !~ /^99+$/ && $nda !~ /^00+$/;
		$ndadrugs{$nda}{$drugname}++;
	}
	close F;
	return \%drugName, \%drugIng, \%drugNDA, \%seenInQ;
}

sub readINDI {
	my($file, $case_drug, $case_ing, $case_nda, $seenInQ) = @_;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		my $caseid = $v[$col{'caseid'}];
		if ($caseid) {
			next unless $seenInQ->{$caseid};
		} else {
			next unless $seenInQ->{$primary};
		}
		my $drugseq = $v[$col{'indi_drug_seq'} // $col{'drug_seq'}];
		my $indication = lc $v[$col{'indi_pt'}];
		$indication =~ s/^\s+//;
		$indication =~ s/\s+$//;
		$indication =~ s/\s\s+/ /g;
		$indication = "drug use for unknown indication" if $indication eq "drug use fo runknown indication";
		next if $ignore{$indication};
		
		my(%ndas, %names);
		if (my $drugnda = $case_nda->{"$primary\$$drugseq"}) {
			$ndas{$drugnda}++;
		}
		my $drugname;
		if ($drugname = $case_drug->{"$primary\$$drugseq"}) {
			$names{$drugname}++;
			if (defined $NDAfor->{$drugname}) {
				foreach my $nda (split /\|/, $NDAfor->{$drugname}) {
					$ndas{$nda}++;
				}
			}
			my $ingredients = $ingIn->{$drugname};
			if (defined $ingredients) {
				foreach my $ingredient (keys %$ingredients) {
					foreach my $nda (split /\|/, $NDAfor->{$ingredient}) {
						$ndas{$nda}++;
					}
				}
			}
		}
		my $ingname;
		if ($ingname = $case_ing->{"$primary\$$drugseq"}) {
			if (defined $NDAfor->{$ingname}) {
				foreach my $nda (split /\|/, $NDAfor->{$ingname}) {
					$ndas{$nda}++;
				}
			}
		}
		
		if (%ndas) {
			foreach my $nda (keys %ndas) {
				$drugNDA_indi{$nda}{$indication}++; ## this would be the place to store case ids, instead of just incrementing
			}
			#print join("\t", $file, $primary, $caseid, $drugseq, $drugname, $ingname, $indication), "\n" if $ingname =~ /ibuprofen/i && $indication =~ /ulcer($|s$| dis)/i;
		} else {
			foreach my $name (keys %names) {
				$drugName_indi{$name}{$indication}++; ## and here
			}
		}
	}
	close F;
}
