/****************************************************************************
* Copyright 2018 by Sensor Lab, Seoul R&D Campus Inc.,
* 33 Seongchon-gil, Seocho-gu, Seoul 06765, Korea.
* All rights reserved.
*
* This software is the confidential and proprietary information
* of Samsung Electronics, Inc. ("Confidential Information").  You
* shall not disclose such Confidential Information and shall use
* it only in accordance with the terms of the license agreement
* you entered into with Samsung.
*/

/**
* @file			SBRMITypes.h
* @brief			header file
* @version		$Revision: 0.1 $
* @date			$Date: 2018-04-19 04:03:28 $
* @author		$Author: sbpark $
*/

#pragma once

#ifndef __SBRMITTYPES_H__
#define __SBRMITTYPES_H__
/************************************** Include ***************************************/


/************************************ User Define ************************************/
#define CONFIGURATION_DLL_TYPE			// used Matlab
//#define CONFIGURATION_ARM_TYPE		// used ARM
//#define CONFIGURATION_SO_TYPE			// used Python

//

#define		ONE_PACKET_SIZE		929		//SNU & SMC : 461,		Elderly Care Robot : 929
#define		NUM_COUNTERS		448		//SNU & SMC : 448,		Elderly Care Robot : 256
#define		OFFSET_IDX			30		//Elderly Care Robot : 50cm		//Initial (25 * 0.4cm = 10 cm ) is ignored

/************************************** Define ***************************************/
/** Boolean TRUE/FALSE type. */
typedef	char							XBOOL;
/** Signed character for strings. */
typedef	char							XCHAR;
/** Unsigned character for strings. */
typedef	unsigned char					XUCHAR;

/** Signed wide character for strings. */
//typedef	wchar_t						XWCHAR;

/** 8-bit signed integer. */
typedef	signed char						XINT8;
/** 8-bit unsigned integer. */
typedef	unsigned char					XUINT8;

/** 16-bit signed integer. */
typedef	signed short					XINT16;
/** 16-bit unsigned integer. */
typedef	unsigned short					XUINT16;

/** 32-bit signed integer. */
typedef	signed int						XINT32;
/** 32-bit unsigned integer. */
typedef	unsigned int					XUINT32;

/** 64-bit signed integer. */
typedef	signed long long				XINT64;
/** 64-bit unsigned integer. */
typedef	unsigned long long				XUINT64;

/** Long signed integer. */
typedef	signed long						XLONG;
/** Long unsigned integer. */
typedef	unsigned long					XULONG;

/** Float (32bit) */
typedef	float							XFLOAT;
/** Double (64bit) */
typedef	float							XDOUBLE; //double						XDOUBLE;

typedef XLONG							XRESULT;

#define TRUE		1
#define FALSE		0

#define	PI_UWB			3.141592653589793
#define	PI_UWB_Mul2		6.283185307179586


#endif /* __SBRMITTYPES_H__ */
