package com.fragma.source;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.Font;
import org.apache.poi.ss.usermodel.IndexedColors;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import com.fragma.model.Deliveries;
import com.fragma.model.Matches;
import com.fragma.model.TeamScores;
import com.fragma.util.FindTopRunRate;
import com.fragma.util.SortingMaps;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

public class Services {
	
	/*---------------------------------------------------------------------
    |  Method getMatches
    |
    |  Purpose:  Read the excel sheet and create list of match Objects
    |  Returns:  List of match objects
    *-------------------------------------------------------------------*/
	public ArrayList<Matches> getMatches() {
		FileReader reader = null;
		try {

			File matchesFile = new File("G:\\Fragma\\ipl\\Data\\resources\\matches.csv");
			reader = new FileReader(matchesFile);
		} catch (FileNotFoundException e1) {

			e1.printStackTrace();
		}

		String pattern = "yyyy-MM-dd";

		SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
		CSVReader csvReader = new CSVReaderBuilder(reader).withSkipLines(1).build();
		String[] row = null;

		ArrayList<Matches> listOfMatches = new ArrayList<Matches>();
		try {
			while ((row = csvReader.readNext()) != null) {
				Matches match = new Matches();
				for (int columnIndex = 0; columnIndex < row.length; columnIndex++) {
					if (columnIndex == 0) {
						match.setMatchId(Integer.parseInt(row[columnIndex]));
					} else if (columnIndex == 1) {
						match.setSeason(Integer.parseInt(row[columnIndex]));
					} else if (columnIndex == 2) {
						match.setCity(row[columnIndex]);
					} else if (columnIndex == 3) {
						try {
							match.setMatchDate(simpleDateFormat.parse(row[columnIndex]));
						} catch (ParseException e) {
							System.err.println("Unable to parse String to date");
							e.printStackTrace();
						}
					} else if (columnIndex == 4) {
						match.setTeam1(row[columnIndex]);
					} else if (columnIndex == 5) {
						match.setTeam2(row[columnIndex]);
					} else if (columnIndex == 6) {
						match.setTossWinner(row[columnIndex]);
					} else if (columnIndex == 7) {
						match.setTossDecision(row[columnIndex]);
					} else if (columnIndex == 8) {
						match.setResult(row[columnIndex]);
					} else if (columnIndex == 9) {
						match.setWinner(row[columnIndex]);
					}
				}
				listOfMatches.add(match);
			}

		} catch (IOException e) {

			e.printStackTrace();
		}

		return listOfMatches;

	}
	
	/*---------------------------------------------------------------------
    |  Method getDeliveries
    |
    |  Purpose:  Read the excel sheet and create list of matches Delivery Objects
    |  Returns:  List of match delivery objects
    *-------------------------------------------------------------------*/
	public ArrayList<Deliveries> getDeliveries() {
		FileReader reader = null;
		try {

			File deliveriesFile = new File("G:\\Fragma\\ipl\\Data\\resources\\deliveries.csv");
			reader = new FileReader(deliveriesFile);
		} catch (FileNotFoundException e1) {

			e1.printStackTrace();
		}
		CSVReader csvReader = new CSVReaderBuilder(reader).withSkipLines(1).build();
		String[] row = null;
		ArrayList<Deliveries> listOfDeliveries = new ArrayList<Deliveries>();
		try {
			while ((row = csvReader.readNext()) != null) {
				Deliveries delivery = new Deliveries();

				for (int columnIndex = 0; columnIndex < row.length; columnIndex++) {
					switch (columnIndex) {
					case 0:
						delivery.setMatch_id(Integer.parseInt(row[columnIndex]));
						break;
					case 1:
						delivery.setInning(Integer.parseInt(row[columnIndex]));
						break;
					case 2:
						delivery.setBattingTeam(row[columnIndex]);
						break;
					case 3:
						delivery.setBowlingTeam(row[columnIndex]);
						break;
					case 4:
						delivery.setOver(Integer.parseInt(row[columnIndex]));
						break;
					case 5:
						delivery.setBall(Integer.parseInt(row[columnIndex]));
						break;
					case 6:
						delivery.setBatsman(row[columnIndex]);
						break;
					case 7:
						delivery.setBowler(row[columnIndex]);
						break;
					case 8:
						delivery.setWideruns(Integer.parseInt(row[columnIndex]));
						break;
					case 9:
						delivery.setByeruns(Integer.parseInt(row[columnIndex]));
						break;
					case 10:
						delivery.setLegbyeruns(Integer.parseInt(row[columnIndex]));
						break;
					case 11:
						delivery.setNoballruns(Integer.parseInt(row[columnIndex]));
						break;
					case 12:
						delivery.setPenaltyruns(Integer.parseInt(row[columnIndex]));
						break;
					case 13:
						delivery.setBatsmanruns(Integer.parseInt(row[columnIndex]));
						break;
					case 14:
						delivery.setExtraruns(Integer.parseInt(row[columnIndex]));
						break;
					case 15:
						delivery.setTotalruns(Integer.parseInt(row[columnIndex]));
						break;

					}

				}
				listOfDeliveries.add(delivery);

			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return listOfDeliveries;

	}
	
	/*---------------------------------------------------------------------
    |  Method fieldFirst
    |
    |  Purpose:  Find the top four teams in different years who chose to field first on winning toss
    |  Parameters:  List of match objects and workbook in order to save the data into excel sheet
    *-------------------------------------------------------------------*/
	public void fieldFirst(ArrayList<Matches> matches, Workbook workbook) {
		int Year = 2017;
		HashMap<String, Integer> hm = fieldsFirstBasedOnYear(Year, matches);

		hm = SortingMaps.sortByValue(hm);
		int top = 0;
		Sheet sheet = workbook.getSheetAt(0);
		Font headerFont = workbook.createFont();
		headerFont.setBold(true);
		headerFont.setFontHeightInPoints((short) 14);
		headerFont.setColor(IndexedColors.RED.getIndex());

		// Create a CellStyle with the font
		CellStyle headerCellStyle = workbook.createCellStyle();
		headerCellStyle.setFont(headerFont);
		int excelRowId = 0;
		// Create a Row
		Row headerRow = sheet.createRow(excelRowId++);

		Cell cell = headerRow.createCell(0);
		cell.setCellValue("Year");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(1);
		cell.setCellValue("Team Name");
		cell.setCellStyle(headerCellStyle);
		cell = headerRow.createCell(2);
		cell.setCellValue("Count");
		cell.setCellStyle(headerCellStyle);

		Row dataRow;
		System.out.println("Year\tTeam Name\t\tCount");
		for (Map.Entry<String, Integer> en : hm.entrySet()) {
			System.out.println(Year + "\t" + en.getKey() + "\t\t" + en.getValue());
			dataRow = sheet.createRow(excelRowId++);
			Cell cell1 = dataRow.createCell(0);
			cell1.setCellValue(Year);

			cell1 = dataRow.createCell(1);
			cell1.setCellValue(en.getKey());

			cell1 = dataRow.createCell(2);
			cell1.setCellValue(en.getValue());

			top++;
			if (top == 4)
				break;
		}
		Year = 2016;
		hm = fieldsFirstBasedOnYear(Year, matches);

		hm = SortingMaps.sortByValue(hm);
		top = 0;
		for (Map.Entry<String, Integer> en : hm.entrySet()) {
			System.out.println(Year + "\t" + en.getKey() + "\t\t" + en.getValue());
			dataRow = sheet.createRow(excelRowId++);
			Cell cell1 = dataRow.createCell(0);
			cell1.setCellValue(Year);

			cell1 = dataRow.createCell(1);
			cell1.setCellValue(en.getKey());

			cell1 = dataRow.createCell(2);
			cell1.setCellValue(en.getValue());
			top++;
			if (top == 4)
				break;
		}
		for (int i = 0; i < 3; i++) {
			sheet.autoSizeColumn(i);
		}

	}
	
	/*---------------------------------------------------------------------
    |  Method teamScoreDetails
    |
    |  Purpose:  Find the number of runs score by each team 
    |  Parameters:  List of match objects and list of deliveries
    |  Returns:  List of Object array containing scores of each team based on Year
    *-------------------------------------------------------------------*/
	public Object[][] teamScoreDetails(ArrayList<Matches> matches, ArrayList<Deliveries> listOfDeliveries) {
		// get team names

		List<String> teams = new ArrayList<String>();
		List<Integer> seasons = new ArrayList<Integer>();

		for (Matches match : matches) {
			teams.add(match.getTeam1());
			teams.add(match.getTeam2());
			seasons.add(match.getSeason());
		}
		// filter the distinct
		List<String> distinctTeamsList = teams.stream().distinct().collect(Collectors.toList());
		// get year
		List<Integer> years = seasons.stream().distinct().collect(Collectors.toList());
		Object[][] teamdetails = new Object[years.size() * distinctTeamsList.size()][4];
		int row = 0;
		for (Integer year : years) {
			for (String teamName : distinctTeamsList) {
				List<Integer> matchIds = new ArrayList<Integer>();
				for (Matches match : matches) {
					if (match.getSeason() == year && (match.getTeam1().equalsIgnoreCase(teamName)
							|| match.getTeam2().equalsIgnoreCase(teamName))) {
						matchIds.add(match.getMatchId());
					}
				}
				matchIds = matchIds.stream().distinct().collect(Collectors.toList());
				TeamScores teamRunScore = teamScore(matchIds, teamName, listOfDeliveries);
				teamdetails[row][0] = year;
				teamdetails[row][1] = teamName;
				teamdetails[row][2] = teamRunScore;
				row++;
			}
		}

		
		return teamdetails;
	}
	/*---------------------------------------------------------------------
    |  Method teamScore
    |
    |  Purpose:  Find the teamScores based on list of matchIds for particular Team 
    |  Parameters:  List of deliveries objects , list of matchId's, team Name
    |  Returns:  teamScores Object
    *-------------------------------------------------------------------*/
	public TeamScores teamScore(List<Integer> matchIdList, String team, ArrayList<Deliveries> listOfDeliveries) {
		TeamScores teamruns = new TeamScores();
		teamruns.setTeamName(team);

		for (Deliveries deliveries : listOfDeliveries) {
			if (matchIdList.contains(deliveries.getMatch_id()) && deliveries.getBattingTeam().equalsIgnoreCase(team)) {
				teamruns.setTotal(teamruns.getTotal() + deliveries.getTotalruns());
				if (deliveries.getBatsmanruns() == 4)
					teamruns.setFours(teamruns.getFours() + 1);
				else if (deliveries.getBatsmanruns() == 6)
					teamruns.setNoofSixes(teamruns.getNoofSixes() + 1);

			}

		}

		return teamruns;

	}
	
	/*---------------------------------------------------------------------
    |  Method fieldsFirstBasedOnYear
    |
    |  Purpose:  Find the number of times the different teams chose to field first based on year 
    |  Parameters:  Year, list of matches
    |  Returns:  Map containing key as team name and count as value
    *-------------------------------------------------------------------*/
	private HashMap<String, Integer> fieldsFirstBasedOnYear(int Year, ArrayList<Matches> matches) {

		List<String> listAllTeams = new ArrayList<String>();

		for (Matches match : matches) {
			listAllTeams.add(match.getTossWinner());
		}
		// filter the distinct
		List<String> distinctList = listAllTeams.stream().distinct().collect(Collectors.toList());
		HashMap<String, Integer> hm = new HashMap<String, Integer>();
		for (String teams : distinctList) {
			hm.put(teams, 0);
		}
		for (Matches match : matches) {
			if (match.getSeason() == Year) {
				if (match.getTossDecision().equalsIgnoreCase("field")) {
					hm.get(match.getTossWinner());
					hm.replace(match.getTossWinner(), hm.get(match.getTossWinner()) + 1);
				}
			}
		}

		return hm;

	}
	
	/*---------------------------------------------------------------------
    |  Method getHighestRunRate
    |
    |  Purpose:  Find the teams which have highest run rate on yearly basis 
    |  Parameters:  list of matches and deliveries
    |  Returns:  Map containing key as year  and team name as value
    *-------------------------------------------------------------------*/
	public HashMap<Integer, String> getHighestRunRate(ArrayList<Matches> matches,
			ArrayList<Deliveries> listOfDeliveries) {
		List<String> teams = new ArrayList<String>();
		List<Integer> seasons = new ArrayList<Integer>();

		for (Matches match : matches) {
			teams.add(match.getTeam1());
			teams.add(match.getTeam2());
			seasons.add(match.getSeason());
		}
		// filter the distinct
		List<String> distinctTeamsList = teams.stream().distinct().collect(Collectors.toList());
		// get year
		List<Integer> years = seasons.stream().distinct().collect(Collectors.toList());
		HashMap<Integer, String> highestRunRates = new HashMap<>();
		for (Integer year : years) {
			HashMap<String, Double> runRatesForYear = new HashMap<>();
			for (String teamName : distinctTeamsList) {
				List<Integer> matchIds = new ArrayList<Integer>();
				for (Matches match : matches) {
					if (match.getSeason() == year && (match.getTeam1().equalsIgnoreCase(teamName)
							|| match.getTeam2().equalsIgnoreCase(teamName))) {
						matchIds.add(match.getMatchId());
					}
				}
				matchIds = matchIds.stream().distinct().collect(Collectors.toList());
				int[] scoreDetails = getScoreDetails(matchIds, teamName, listOfDeliveries);
				runRatesForYear.put(teamName,
						netRunRate(scoreDetails[0], scoreDetails[1], scoreDetails[2], scoreDetails[3]));

			}

			highestRunRates.put(year, FindTopRunRate.sortByValue(runRatesForYear));
		}
		return highestRunRates;
	}
	
	/*---------------------------------------------------------------------
    |  Method netRunRate
    |
    |  Purpose:  Calculate the net run rate 
    |  Parameters:  Runs Scored,Overs Faced, Runs Conceded, Overs Bowled
    |  Returns:  net run rate
    *-------------------------------------------------------------------*/
	private double netRunRate(int runsScored, int OversFaced, int runsConceded, int OversBowled) {
		double netRunRate = 0;
		if (OversBowled != 0 && OversFaced != 0) {
			netRunRate = (runsScored / OversFaced) - (runsConceded / OversBowled);
		}
		return netRunRate;

	}
	
	/*---------------------------------------------------------------------
    |  Method getScoreDetails
    |
    |  Purpose:  Calculate the runs Scored,Overs Faced, Runs Conceded, Overs Bowled for list of matches a team has played in particular year
    |  Parameters:  matchId's list, team name, list of deliveries
    |  Returns:  array of score details
    *-------------------------------------------------------------------*/
	private int[] getScoreDetails(List<Integer> matchIdList, String team, ArrayList<Deliveries> listOfDeliveries) {
		int[] oversDetail = new int[4];

		int runsScored = 0;
		int OversFaced = 0;
		int runsConceded = 0;
		int OversBowled = 0;
		for (Deliveries deliveries : listOfDeliveries) {
			if (matchIdList.contains(deliveries.getMatch_id()) && deliveries.getBattingTeam().equalsIgnoreCase(team)) {
				runsScored = runsScored + deliveries.getTotalruns();
				OversFaced = Math.max(OversFaced, deliveries.getOver());
			}
		}
		for (Deliveries deliveries : listOfDeliveries) {
			if (matchIdList.contains(deliveries.getMatch_id()) && deliveries.getBowlingTeam().equalsIgnoreCase(team)) {
				runsConceded = runsConceded + deliveries.getTotalruns();
				OversBowled = Math.max(OversBowled, deliveries.getOver());
			}
		}
		oversDetail[0] = runsScored;
		oversDetail[1] = OversFaced;
		oversDetail[2] = runsConceded;
		oversDetail[3] = OversBowled;

		return oversDetail;

	}
	
	/*---------------------------------------------------------------------
    |  Method bestEconomyBowlers
    |
    |  Purpose:  Get best Economy bowlers (Top 10)
    |  Parameters:  list of matches and list of deliveries
    |  Returns:  array of Object containing best economy bowleres along with economy and year
    *-------------------------------------------------------------------*/
	public Object[][] bestEconomyBowlers(ArrayList<Matches> matches, ArrayList<Deliveries> listOfDeliveries) {

		List<Integer> seasons = new ArrayList<Integer>();

		for (Matches match : matches) {
			seasons.add(match.getSeason());
		}
		// get year
		List<Integer> years = seasons.stream().distinct().collect(Collectors.toList());
		Object[][] bestEconomyBowlers = new Object[years.size() * 10][3];
		int row = 0;
		for (Integer year : years) {
			List<Integer> matchIds = new ArrayList<Integer>();
			for (Matches match : matches) {
				if (match.getSeason() == year) {
					matchIds.add(match.getMatchId());
				}
			}
			matchIds = matchIds.stream().distinct().collect(Collectors.toList());
			HashMap<String, Double> bowlersEconomy = getBowlersDetails(matchIds, listOfDeliveries);
			HashMap<String, Double> bowlersEconomySorted = SortingMaps.sortByDoubleValue(bowlersEconomy);
			int count = 0;

			for (Map.Entry<String, Double> mapEntry : bowlersEconomySorted.entrySet()) {
				if (count < 10) {
					bestEconomyBowlers[row][0] = year;
					bestEconomyBowlers[row][1] = mapEntry.getKey();
					bestEconomyBowlers[row][2] = mapEntry.getValue();
					row++;
				}
				count++;
			}

		}
		return bestEconomyBowlers;

	}
	
	/*---------------------------------------------------------------------
    |  Method getBowlersDetails
    |
    |  Purpose:  Get bowlers economy based on matches bowled
    |  Parameters:  list of MatchId's and deliveries
    |  Returns:  Map containing bowlers along with economy
    *-------------------------------------------------------------------*/
	private HashMap<String, Double> getBowlersDetails(List<Integer> matchIdList,
			ArrayList<Deliveries> listOfDeliveries) {
		List<String> listOfBowlers = new ArrayList<String>();
		for (Deliveries deliveries : listOfDeliveries) {
			if (matchIdList.contains(deliveries.getMatch_id())) {
				listOfBowlers.add(deliveries.getBowler());
			}
		}
		listOfBowlers = listOfBowlers.stream().distinct().collect(Collectors.toList());
		HashMap<String, Double> bowlersEconomy = new HashMap<String, Double>();
		for (String bowler : listOfBowlers) {
			int runsGiven = 0;
			int noofballs = 0;
			int noofOvers = 0;
			for (Deliveries deliveries : listOfDeliveries) {
				if (matchIdList.contains(deliveries.getMatch_id()) && bowler.equalsIgnoreCase(deliveries.getBowler())) {
					if (deliveries.getLegbyeruns() == 0 && deliveries.getByeruns() == 0) {
						runsGiven = runsGiven + deliveries.getTotalruns();
					}
					if(deliveries.getBall()<=6) {
					noofballs++;
					}
					if (noofballs == 6) {
						noofOvers++;
						noofballs = 0;
					}

				}
			}
			if (noofOvers > 10) {
				bowlersEconomy.put(bowler, getEconomy(runsGiven, noofOvers));
			}
		}
		return bowlersEconomy;
	}
	
	/*---------------------------------------------------------------------
    |  Method getEconomy
    |
    |  Purpose:  Calculate economy of Bowleres
    |  Parameters:  runs given and overs bowled
    |  Returns:  economy
    *-------------------------------------------------------------------*/
	private double getEconomy(int runsGiven, int OversBowled) {
		if (OversBowled != 0) {
			return runsGiven / OversBowled;
		} else
			return 0;

	}

}
