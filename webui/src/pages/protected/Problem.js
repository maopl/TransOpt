import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Problems from '../../features/problems/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Specify Problems"}))
      }, [])


    return(
        <Problems />
    )
}

export default InternalPage